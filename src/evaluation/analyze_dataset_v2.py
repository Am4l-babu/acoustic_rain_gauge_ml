import os
import pandas as pd
import numpy as np
from pathlib import Path
import soundfile as sf
from datetime import datetime, timedelta
import json
from collections import defaultdict
import re

# Configuration
SOURCE_DRIVE = Path(r"F:\arg_dataset_unzip")
REPORT_FILE = Path(r"D:\acoustic_rain_gauge_ml\docs\dataset_analysis_report_v2.txt")


def find_all_files_recursive(folder_path):
    """Find ALL files recursively in subdirectories"""
    all_files = {
        'audio_like': [],
        'video_like': [],
        'data_files': [],
        'other': []
    }

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()

            if ext in ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']:
                all_files['audio_like'].append(file_path)
            elif ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.3gp']:
                all_files['video_like'].append(file_path)
            elif ext in ['.csv', '.xlsx', '.xls', '.json', '.txt']:
                all_files['data_files'].append(file_path)
            else:
                all_files['other'].append(file_path)

    return all_files


def analyze_folder_structure(folder_path):
    """Deep analysis of folder including subdirectories"""
    results = {
        'folder': folder_path.name,
        'total_files': 0,
        'audio_files': [],
        'video_files': [],
        'data_files': [],
        'files_without_ext': [],
        'subdirectories': [],
        'mech_data': None,
        'audio_duration_stats': {},
        'file_size_mb': 0
    }

    all_files = find_all_files_recursive(folder_path)

    results['audio_files'] = all_files['audio_like']
    results['video_files'] = all_files['video_like']
    results['data_files'] = all_files['data_files']
    results['files_without_ext'] = [f for f in all_files['other'] if f.suffix == '']
    results['total_files'] = (len(results['audio_files']) +
                              len(results['video_files']) +
                              len(results['data_files']))

    results['subdirectories'] = [d.name for d in folder_path.iterdir() if d.is_dir()]

    # Analyze mechanical data
    if results['data_files']:
        for data_file in results['data_files']:
            if data_file.suffix == '.csv':
                try:
                    df = pd.read_csv(data_file, nrows=100)
                    results['mech_data'] = {
                        'file': str(data_file),
                        'columns': list(df.columns),
                        'total_rows': len(pd.read_csv(data_file)),
                        'sample': df.head(3).to_dict()
                    }
                    break
                except Exception:
                    continue

    # Sample audio files for duration analysis (soundfile: header-only, no decoding)
    if results['audio_files']:
        durations = []
        sample_rates = []

        for audio_file in results['audio_files'][:20]:
            try:
                info = sf.info(str(audio_file))
                durations.append(info.duration)
                sample_rates.append(info.samplerate)
                results['file_size_mb'] += audio_file.stat().st_size / (1024 * 1024)
            except Exception as e:
                print(f"    Error reading {audio_file.name}: {e}")

        if durations:
            results['audio_duration_stats'] = {
                'count': len(durations),
                'min_sec': float(min(durations)),
                'max_sec': float(max(durations)),
                'mean_sec': float(np.mean(durations)),
                'median_sec': float(np.median(durations)),
                'sample_rates': list(set(sample_rates))
            }

            dur_dist = defaultdict(int)
            for d in durations:
                if d < 5:
                    dur_dist['<5s'] += 1
                elif d < 10:
                    dur_dist['5-10s'] += 1
                elif d < 15:
                    dur_dist['10-15s'] += 1
                else:
                    dur_dist['>15s'] += 1
            results['audio_duration_stats']['distribution'] = dict(dur_dist)

    # Estimate total size for video files
    if results['video_files']:
        for vid_file in results['video_files'][:20]:
            results['file_size_mb'] += vid_file.stat().st_size / (1024 * 1024)

    return results


def generate_detailed_report():
    """Generate comprehensive report with recursive search"""
    folders = [f for f in SOURCE_DRIVE.iterdir() if f.is_dir()]

    report = []
    report.append("=" * 80)
    report.append("ACOUSTIC RAINFALL DATASET ANALYSIS REPORT - VERSION 2")
    report.append("(With Recursive Subdirectory Search)")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Source: {SOURCE_DRIVE}")
    report.append("=" * 80)
    report.append("")

    total_audio = 0
    total_video = 0
    total_data_files = 0
    total_size_gb = 0
    all_durations = []

    folder_results = []

    print(f"Analyzing {len(folders)} folders (including subdirectories)...")

    for i, folder in enumerate(folders):
        print(f"\n[{i+1}/{len(folders)}] {folder.name}")

        result = analyze_folder_structure(folder)
        folder_results.append(result)

        total_audio += len(result['audio_files'])
        total_video += len(result['video_files'])
        total_data_files += len(result['data_files'])
        total_size_gb += result['file_size_mb'] / 1024

        if result['audio_duration_stats']:
            if 'mean_sec' in result['audio_duration_stats']:
                all_durations.append(result['audio_duration_stats']['mean_sec'])

        print(f"  Audio files: {len(result['audio_files'])}")
        print(f"  Video files: {len(result['video_files'])}")
        print(f"  Data files:  {len(result['data_files'])}")
        print(f"  Subdirs:     {len(result['subdirectories'])}")

        if result['audio_duration_stats']:
            print(f"  Audio duration: {result['audio_duration_stats'].get('mean_sec', 'N/A'):.1f}s avg")
            if 'distribution' in result['audio_duration_stats']:
                print(f"  Distribution: {result['audio_duration_stats']['distribution']}")

        if result['mech_data']:
            print(f"  Mechanical data: {result['mech_data']['total_rows']} rows")
            print(f"  Columns: {result['mech_data']['columns']}")

    # OVERALL SUMMARY
    report.append("OVERALL DATASET SUMMARY (RECURSIVE SEARCH)")
    report.append("-" * 80)
    report.append(f"Total folders analyzed: {len(folders)}")
    report.append(f"Total audio files found: {total_audio:,}")
    report.append(f"Total video files found: {total_video:,}")
    report.append(f"Total data files found:  {total_data_files:,}")
    report.append(f"Estimated total size:    {total_size_gb:.2f} GB")
    report.append("")

    # AUDIO ANALYSIS
    if all_durations:
        report.append("AUDIO DURATION ANALYSIS")
        report.append("-" * 80)
        report.append(f"Average duration: {np.mean(all_durations):.2f} seconds")
        report.append(f"Duration range:   {min(all_durations):.2f}s - {max(all_durations):.2f}s")
        report.append("")

    # DETAILED FOLDER BREAKDOWN
    report.append("DETAILED FOLDER BREAKDOWN")
    report.append("-" * 80)

    for result in folder_results:
        report.append(f"\n{result['folder']}:")
        report.append(f"  Audio: {len(result['audio_files'])} files")
        report.append(f"  Video: {len(result['video_files'])} files")
        report.append(f"  Data:  {len(result['data_files'])} files")
        report.append(f"  Subdirs: {len(result['subdirectories'])}")

        if result['subdirectories']:
            subdir_preview = ', '.join(result['subdirectories'][:5])
            suffix = '...' if len(result['subdirectories']) > 5 else ''
            report.append(f"  Subdir names: {subdir_preview}{suffix}")

        if result['mech_data']:
            report.append(f"  Mech columns: {result['mech_data']['columns']}")

        if result['audio_duration_stats'] and 'distribution' in result['audio_duration_stats']:
            report.append(f"  Duration dist: {result['audio_duration_stats']['distribution']}")

    # RECOMMENDATIONS
    report.append("\n" + "=" * 80)
    report.append("CRITICAL FINDINGS & RECOMMENDATIONS")
    report.append("=" * 80)

    if total_audio == 0 and total_video > 0:
        report.append("\nAUDIO FILES MIGHT BE VIDEOS!")
        report.append(f"   Found {total_video} video files - these might contain the audio")
        report.append("   -> Consider extracting audio from video files using ffmpeg")
        report.append("   -> Or check if the video files are actually audio with wrong extension")

    if total_audio > 0:
        report.append(f"\nFOUND {total_audio} AUDIO FILES!")
        report.append("   -> Update cleaning script to search recursively")
        report.append("   -> Process subdirectories individually")

    if total_video > 0:
        report.append(f"\nFOUND {total_video} VIDEO FILES")
        report.append("   -> These might be recordings with embedded audio")
        report.append("   -> Use soundfile or ffmpeg to extract audio tracks")

    report.append("\nNEXT STEPS:")
    report.append("   1. Review this report to understand file structure")
    report.append("   2. Decide if video files need audio extraction")
    report.append("   3. Update cleaning script to handle nested folders")
    report.append("   4. Process folder-by-folder to save memory")

    report_text = "\n".join(report)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Report saved to: {REPORT_FILE}")
    print(f"{'='*80}")

    return report_text


if __name__ == "__main__":
    generate_detailed_report()
