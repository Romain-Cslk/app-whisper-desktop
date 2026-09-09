"""Do not let missing Qt or Windows DPAPI silently count as a validated release."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED = {
    'test_widget_merge_undo_split_and_immediate_preview',
    'test_widget_cancel_never_calls_apply_and_clears_memory',
    'test_widget_apply_passes_partition_and_revision_guards',
    'test_widget_biometric_consent_declined_does_not_apply',
    'test_widget_apply_failure_stays_open_and_can_retry',
    'test_widget_switching_excerpts_and_closing_stops_player',
    'test_missing_file_is_explicit_and_no_playback_starts',
    'test_real_wav_load_seek_and_auto_stop',
    'test_windows_dpapi_real_roundtrip',
}


def verify(path: Path) -> None:
    cases = ET.parse(path).getroot().iter('testcase')
    executed = set()
    for case in cases:
        name = case.get('name', '')
        if name in REQUIRED:
            if any(case.find(tag) is not None for tag in ('skipped', 'error', 'failure')):
                raise RuntimeError(f'Required Windows/Qt test was skipped or failed: {name}')
            executed.add(name)
    missing = REQUIRED - executed
    if missing:
        raise RuntimeError('Required Windows/Qt tests did not execute: ' + ', '.join(sorted(missing)))
    print(f'{len(REQUIRED)} mandatory Qt/Windows tests executed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('junit_xml', type=Path)
    verify(parser.parse_args().junit_xml)
