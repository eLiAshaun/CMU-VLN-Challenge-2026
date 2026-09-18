"""Shared RGB-D timestamp checks for capture and read-only preflight."""
import math


def packet_timing(messages, now, max_age, max_offset):
    stamps = [float(m.header.stamp.sec) + float(m.header.stamp.nanosec)*1e-9 for m in messages]
    ages = [now-value for value in stamps]
    spread = max(stamps)-min(stamps)
    return {
        'timestamps_finite': all(math.isfinite(value) for value in stamps),
        'all_streams_fresh': all(-0.1 <= value <= max_age for value in ages),
        'streams_synchronized': spread <= max_offset + 1e-9,
        'stamp_spread_seconds': spread,
        'stream_ages_seconds': ages,
    }


def timing_valid(result):
    return all(result[key] for key in ('timestamps_finite', 'all_streams_fresh', 'streams_synchronized'))
