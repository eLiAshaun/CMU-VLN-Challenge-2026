"""Compositional English frontend for the competition's three task families.

Object names are open vocabulary. Syntax, modifier scope, returned entities and
route actions are constructed by the parser; no question/scene/answer lookup is
used. The physical TaskIR interpreter remains the single runtime executor.
"""
from __future__ import annotations
import re

_COLORS={'red','blue','black','white','green','yellow','orange','purple','pink','brown','gray','grey'}
_SIZES={'small','large','big','tiny'}
_ARTICLES={'the','a','an','two'}
_RELATIONS={'on','inside','in','near','above','below','under','between','with','closest','nearest','farthest','furthest','that','which'}


def _singular(word):
    if word.endswith('ies'):return word[:-3]+'y'
    if word.endswith(('ches','shes','xes','zes')):return word[:-2]
    if word.endswith('s') and not word.endswith(('ss','us','is')) and word not in ('stairs',):return word[:-1]
    return word


def _class(words):
    words=[w for w in words if w not in _ARTICLES]
    if not words:raise ValueError('object category missing')
    attrs=[]
    while words and words[0] in _COLORS|_SIZES:
        word=words.pop(0);attrs.append(('color' if word in _COLORS else 'size',word))
    if not words:raise ValueError('attribute without an object category')
    words[-1]=_singular(words[-1])
    name=' '.join(words)
    result={'op':'filter_class','class':name,'visual_class':name}
    for key,value in attrs:result={'op':'filter_attribute','source':result,'attribute':key,'value':value}
    return result


def _relation(op,source,anchor,inverse=False):
    return {'op':op,'subject':anchor if inverse else source,
            'anchor':source if inverse else anchor,
            'return_role':'anchor' if inverse else 'subject'}


class _Parser:
    def __init__(self,text):
        self.words=re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[,.;]",text.lower())
        self.i=0

    def peek(self,offset=0):return self.words[self.i+offset] if self.i+offset<len(self.words) else None
    def take(self):
        value=self.peek()
        if value is None:raise ValueError('unexpected end of instruction')
        self.i+=1;return value
    def consume(self,*words):
        if self.words[self.i:self.i+len(words)]==list(words):self.i+=len(words);return True
        return False
    def skip_articles(self):
        while self.peek() in _ARTICLES:self.i+=1
    def end(self):return self.peek() is None or self.peek() in ('.',';')

    def description(self,stop=frozenset(),outer=None):
        self.skip_articles()
        start=self.i
        while (self.peek() is not None and self.peek() not in _RELATIONS|set(stop)|{',','.',';','and','then','to','at','by','avoiding','avoid','is','are'}):
            self.i+=1
        node=_class(self.words[start:self.i])
        self.consume('is');self.consume('are')
        while not self.end() and self.peek() not in set(stop)|{',','and','then','to','at','by','avoiding','avoid'}:
            if self.consume('that') or self.consume('which'):
                self.consume('is');self.consume('are')
                if self.peek() not in ('has','have','with') and self.peek() not in _RELATIONS:
                    raise ValueError('unsupported relative clause')
            word=self.peek()
            if word in ('has','have','with'):
                self.i+=1
                physical={'on','inside','in','above','below','under','near'}
                item=self.description(stop=stop|physical,outer=node)
                if self.peek() in physical:
                    relation=self.take()
                    if self.peek() not in ('it','them'):raise ValueError('relative relation requires its owner')
                    self.i+=1;node=_relation({'in':'inside','under':'below'}.get(relation,relation),node,item,True)
                else:
                    raise ValueError('with clause must state its physical relation')
            elif word in ('closest','nearest','farthest','furthest'):
                self.i+=1
                if not (self.consume('to') or self.consume('from')):raise ValueError('distance comparison anchor missing')
                anchor=self.description(stop=stop,outer=outer)
                node={'op':'argmin_distance' if word in ('closest','nearest') else 'argmax_distance','candidates':node,'anchor':anchor}
            elif word=='between':
                self.i+=1
                anchors=self.anchor_pair(stop)
                node={'op':'between','subject':node,'anchors':anchors}
            elif word in ('on','inside','in','near','above','below','under'):
                # A pronoun closes a relation owned by an outer description;
                # it does not introduce a new detectable object named "it".
                if self.peek(1) in ('it','them') and outer is not None:break
                self.i+=1
                anchor=self.description(stop=stop,outer=outer)
                node=_relation({'in':'inside','under':'below'}.get(word,word),node,anchor)
            else:break
        return node

    def anchor_pair(self,stop=frozenset()):
        dual=False
        look=self.i
        while look<len(self.words) and self.words[look] in _ARTICLES:
            dual|=self.words[look]=='two';look+=1
        left=self.description(stop=stop|{'and'})
        if self.consume('and'):return [left,self.description(stop=stop)]
        if dual:return [left,left]
        raise ValueError('between requires two anchor descriptions')

    def separators(self):
        while self.peek() in (',','and','then','first','finally'):self.i+=1

    def route(self):
        steps=[]
        while not self.end():
            self.separators()
            if self.end():break
            if self.consume('avoiding') or self.consume('avoid'):
                self.consume('the');self.consume('path')
                if self.consume('between'):
                    steps.append({'action':'avoid_region','region':{'op':'between_region','anchors':self.anchor_pair()}})
                else:
                    self.consume('near');steps.append({'action':'avoid_path','target':self.description()})
                continue
            if self.consume('take') or self.consume('follow'):
                self.consume('the')
                if not self.consume('path'):raise ValueError('path action missing path')
                if self.consume('between'):steps.append({'action':'between_path','anchors':self.anchor_pair()})
                elif self.consume('near'):steps.append({'action':'near_path','target':self.description()})
                else:raise ValueError('path action requires near or between')
                continue
            if self.consume('pass'):
                if not (self.consume('by') or self.consume('near')):raise ValueError('pass requires by or near')
                steps.append({'action':'pass_near','target':self.description()});continue
            if self.consume('stop'):
                if not (self.consume('at') or self.consume('by')):raise ValueError('stop destination missing')
                steps.append({'action':'go_to','target':self.description()});continue
            moved=self.consume('go') or self.consume('walk') or self.consume('navigate')
            if moved and self.consume('between'):
                steps.append({'action':'between_path','anchors':self.anchor_pair()});continue
            if moved and self.consume('near'):
                steps.append({'action':'pass_near','target':self.description()});continue
            if self.consume('to'):
                steps.append({'action':'go_to','target':self.description()});continue
            raise ValueError('unsupported movement syntax near: '+' '.join(self.words[self.i:self.i+8]))
        for index,step in enumerate(steps):step['order']=index
        if not steps:raise ValueError('instruction has no route')
        return {'steps':steps}

    def parse(self):
        if self.consume('how','many'):
            program={'expression':{'op':'count_distinct','source':self.description()}}
        elif self.consume('count'):
            self.consume('the');self.consume('number');self.consume('of')
            program={'expression':{'op':'count_distinct','source':self.description()}}
        elif self.peek() in ('first','go','walk','navigate','take','follow','pass','stop','avoid'):
            program=self.route()
        else:
            self.consume('find');self.consume('identify');self.consume('select')
            program={'expression':self.description()}
        while self.peek() in ('.',';'):self.i+=1
        if self.peek() is not None:raise ValueError('unconsumed task syntax: '+' '.join(self.words[self.i:]))
        return program


def parse_question(question):
    if not isinstance(question,str) or not question.strip():raise ValueError('question must be nonempty')
    return _Parser(question).parse()
