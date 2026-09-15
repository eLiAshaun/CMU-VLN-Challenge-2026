"""Current-image membership of an independently identified physical artifact."""
import json

MEMBERSHIP_SCHEMA={
    'title':'same_artifact_visual_membership_v1','type':'object',
    'properties':{'verdict':{'type':'string','enum':['yes','no','unknown']}},
    'required':['verdict'],'additionalProperties':False,
}


def membership_prompt(identity,requested):
    owner={k:identity[k] for k in ('physical_category','visible_evidence')}
    return (
        'Inspect the physical artifact marked by the red rectangle in this image. '
        'The independent description below identifies the artifact owner; now use the '
        'actual image to decide whether that SAME artifact belongs to the requested category. '
        'Judge its visible physical structure and established function, not merely the '
        'similarity of two category names. A specific type can belong to a broader category, '
        'but sharing a broad parent, a flat surface, or a possible use does not establish '
        'a specialized type. Do not classify a depicted image, an integral part, or a '
        'neighboring object as a different physical artifact. Use unknown when the '
        'visible features do not establish the requested type. '
        'Return only {"verdict":"yes|no|unknown"}.\n'
        'Independent artifact description: '+json.dumps(owner)+'\nRequested category: '+requested
    )
