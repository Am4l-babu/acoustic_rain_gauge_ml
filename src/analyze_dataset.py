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
REPORT_FILE = Path(r"D:\acoustic_rain_gauge_ml\docs\dataset_analysis_report.txt")


def analyze_mechanical_data(folder_path):
    """Analyze mechanical rainfall data in a folder"""
    results = {
        'folder': folder_path.name,
        'has_mech_data': False,
        'mech_files': [],
        'total_records': 0,
        'time_range': None,
        'time_gaps': [],
        'rainfall_stats': {},
        'column_names': [],
        'errors': []
    }

    # Find CSV files
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        return results

    results['has_mech_data'] = True
    results['mech_files'] = [f.name for f in csv_files]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            results['total_records'] += len(df)
            results['column_names'] = list(df.columns)

            # Find time column
            time_col = None
            for col in df.columns:
                if 'time' in col.lower():
                    time_col = col
                    break

            # Find rainfall column
            rain_col = None
            for col in df.columns:
                if 'rain' in col.lower() or 'payload' in col.lower():
                    rain_col = col
                    break

            if time_col and rain_col:
                # Parse timestamps
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col])

                # Time range
                time_min = df[time_col].min()
                time_max = df[time_col].max()
                results['time_range'] = (time_min, time_max)

                # Detect time gaps (> 5 minutes between readings)
                df_sorted = df.sort_values(time_col)
                time_diffs = df_sorted[time_col].diff()
                large_gaps = time_diffs[time_diffs > timedelta(minutes=5)]
                results['time_gaps'] = len(large_gaps)

                # Rainfall statistics
                rainfall_values = df[rain_col].dropna()
                results['rainfall_stats'] = {
                    'total_tips': int((rainfall_values > 0).sum()),
                    'zero_readings': int((rainfall_values == 0).sum()),
                    'min_value': float(rainfall_values.min()),
                    'max_value': float(rainfall_values.max()),
                    'unique_values': sorted(rainfall_values.unique().tolist())[:10]
                }

        except Exception as e:
            results['errors'].append(f"{csv_file.name}: {str(e)}")

    return results


def analyze_audio_data(folder_path):
    """Analyze audio files in a folder"""
    results = {
        'folder': folder_path.name,
        'has_audio': False,
        'total_files': 0,
        'duration_stats': {},
        'sample_rates': {},
        'file_size_mb': 0,
        'errors': [],
        'duration_distribution': defaultdict(int)
    }

    audio_files = list(folder_path.glob("*.wav"))

    if not audio_files:
        return results

    results['has_audio'] = True
    results['total_files'] = len(audio_files)

    durations = []
    sample_rates_list = []

    for i, audio_file in enumerate(audio_files[:100]):  # Sample first 100 files
        try:
            info = sf.info(str(audio_file))  # header-only read, no decoding
            duration = info.duration
            sr = info.samplerate
            file_size = audio_file.stat().st_size / (1024 * 1024)  # MB

            durations.append(duration)
            sample_rates_list.append(sr)
            results['file_size_mb'] += file_size

            # Categorize duration
            if duration < 5:
                results['duration_distribution']['<5s'] += 1
            elif duration < 10:
                results['duration_distribution']['5-10s'] += 1
            elif duration < 15:
                results['duration_distribution']['10-15s'] += 1
            else:
                results['duration_distribution']['>15s'] += 1

        except Exception as e:
            results['errors'].append(f"{audio_file.name}: {str(e)}")

    if durations:
        results['duration_stats'] = {
            'min_sec': float(min(durations)),
            'max_sec': float(max(durations)),
            'mean_sec': float(np.mean(durations)),
            'median_sec': float(np.median(durations)),
            'std_sec': float(np.std(durations))
        }

        duration_counts = defaultdict(int)
        for d in durations:
            rounded = round(d)
            duration_counts[f"{rounded}s"] += 1
        results['common_durations'] = dict(
            sorted(duration_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )

    if sample_rates_list:
        results['sample_rates'] = {
            'unique_rates': list(set(sample_rates_list)),
            'most_common': max(set(sample_rates_list), key=sample_rates_list.count)
        }

    # Estimate total size
    if len(audio_files) > 100:
        avg_size = results['file_size_mb'] / 100
        results['estimated_total_size_mb'] = avg_size * len(audio_files)
    else:
        results['estimated_total_size_mb'] = results['file_size_mb']

    return results


def generate_report():
    """Generate comprehensive dataset analysis report"""
    folders = [f for f in SOURCE_DRIVE.iterdir() if f.is_dir()]

    report = []
    report.append("=" * 80)
    report.append("ACOUSTIC RAINFALL DATASET ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Source: {SOURCE_DRIVE}")
    report.append("=" * 80)
    report.append("")

    total_folders = len(folders)
    folders_with_audio = 0
    folders_with_mech = 0
    total_audio_files = 0
    total_mech_records = 0
    total_size_gb = 0

    all_mech_results = []
    all_audio_results = []

    print(f"Analyzing {total_folders} folders...")

    for i, folder in enumerate(folders):
        print(f"  [{i+1}/{total_folders}] {folder.name}")

        mech_result = analyze_mechanical_data(folder)
        audio_result = analyze_audio_data(folder)

        all_mech_results.append(mech_result)
        all_audio_results.append(audio_result)

        if mech_result['has_mech_data']:
            folders_with_mech += 1
            total_mech_records += mech_result['total_records']

        if audio_result['has_audio']:
            folders_with_audio += 1
            total_audio_files += audio_result['total_files']
            total_size_gb += audio_result['estimated_total_size_mb'] / 1024

    # OVERALL SUMMARY
    report.append("OVERALL DATASET SUMMARY")
    report.append("-" * 80)
    report.append(f"Total folders: {total_folders}")
    report.append(f"Folders with mechanical data: {folders_with_mech}")
    report.append(f"Folders with audio data: {folders_with_audio}")
    report.append(f"Total mechanical records: {total_mech_records:,}")
    report.append(f"Total audio files: {total_audio_files:,}")
    report.append(f"Estimated total size: {total_size_gb:.2f} GB")
    report.append("")

    # AUDIO ANALYSIS DETAILS
    report.append("AUDIO DATA ANALYSIS")
    report.append("-" * 80)

    all_durations = []
    duration_cats = defaultdict(int)
    sample_rates = []

    for result in all_audio_results:
        if result['has_audio']:
            if result['duration_stats']:
                all_durations.append(result['duration_stats']['mean_sec'])
            for cat, count in result['duration_distribution'].items():
                duration_cats[cat] += count
            if result['sample_rates']:
                sample_rates.extend(result['sample_rates'].get('unique_rates', []))

    report.append(f"Audio duration distribution:")
    for cat, count in sorted(duration_cats.items()):
        report.append(f"  {cat}: {count} files")

    if all_durations:
        report.append(f"\nAverage audio duration: {np.mean(all_durations):.2f} seconds")
        report.append(f"Duration range: {min(all_durations):.2f}s - {max(all_durations):.2f}s")

    if sample_rates:
        report.append(f"\nSample rates found: {set(sample_rates)}")

    report.append("")

    # MECHANICAL DATA ANALYSIS
    report.append("MECHANICAL DATA ANALYSIS")
    report.append("-" * 80)

    all_rainfall_values = []
    total_gaps = 0

    for result in all_mech_results:
        if result['has_mech_data']:
            total_gaps += result['time_gaps']
            if result['rainfall_stats']:
                all_rainfall_values.extend(result['rainfall_stats'].get('unique_values', []))

    report.append(f"Total time gaps (>5 min): {total_gaps}")
    report.append(f"Unique rainfall values: {sorted(set(all_rainfall_values))}")
    report.append("")

    # FOLDER-BY-FOLDER BREAKDOWN
    report.append("FOLDER-BY-FOLDER BREAKDOWN")
    report.append("-" * 80)

    for i, (mech, audio) in enumerate(zip(all_mech_results, all_audio_results)):
        report.append(f"\n{i+1}. {mech['folder']}")

        if mech['has_mech_data']:
            report.append(f"   Mechanical: {mech['total_records']:,} records")
            if mech['time_range']:
                report.append(f"   Time range: {mech['time_range'][0]} to {mech['time_range'][1]}")
            if mech['time_gaps'] > 0:
                report.append(f"   WARNING  Time gaps: {mech['time_gaps']}")
            if mech['rainfall_stats']:
                report.append(f"   Rain tips: {mech['rainfall_stats']['total_tips']}")
        else:
            report.append(f"   WARNING  NO MECHANICAL DATA")

        if audio['has_audio']:
            report.append(f"   Audio: {audio['total_files']} files, {audio['estimated_total_size_mb']:.1f} MB")
            if audio['duration_stats']:
                report.append(f"   Duration: {audio['duration_stats']['mean_sec']:.1f}s avg")
            if audio['duration_distribution']:
                dur_str = ", ".join([f"{k}={v}" for k, v in audio['duration_distribution'].items()])
                report.append(f"   Distribution: {dur_str}")
        else:
            report.append(f"   WARNING  NO AUDIO DATA")

        if mech['errors']:
            report.append(f"   Errors: {len(mech['errors'])}")
        if audio['errors']:
            report.append(f"   Audio errors: {len(audio['errors'])}")

    # RECOMMENDATIONS
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS & ACTION ITEMS")
    report.append("=" * 80)

    recommendations = []

    if duration_cats:
        if len(duration_cats) > 1:
            recommendations.append("WARNING  VARIABLE AUDIO DURATIONS DETECTED")
            recommendations.append("   -> Update cleaning script to handle variable-length audio files")
            recommendations.append("   -> Consider resampling or padding to fixed duration")

    if total_gaps > 0:
        recommendations.append(f"WARNING  {total_gaps} TIME GAPS in mechanical data")
        recommendations.append("   -> Investigate gaps - may indicate sensor malfunction")
        recommendations.append("   -> Decide whether to interpolate or drop these periods")

    if folders_with_audio != folders_with_mech:
        mismatched = abs(folders_with_audio - folders_with_mech)
        recommendations.append(f"WARNING  {mismatched} folders have mismatched data types")
        recommendations.append("   -> Some folders have audio but no mechanical data (or vice versa)")
        recommendations.append("   -> These folders should be excluded or processed separately")

    if set(all_rainfall_values) != {0.0, 0.2}:
        recommendations.append("WARNING  UNEXPECTED RAINFALL VALUES")
        recommendations.append(f"   -> Found values: {sorted(set(all_rainfall_values))}")
        recommendations.append("   -> Verify if these are valid or sensor errors")

    recommendations.append("\nOK CLEANING STRATEGY:")
    recommendations.append("   1. Process folder-by-folder to manage memory")
    recommendations.append("   2. Align audio to mechanical data using timestamps")
    recommendations.append("   3. Handle variable audio durations (3s, 10s, etc.)")
    recommendations.append("   4. Flag or remove time gaps > 5 minutes")
    recommendations.append("   5. Save cleaned data to C: drive")
    recommendations.append("   6. Delete processed folders from F: to free space")

    for rec in recommendations:
        report.append(rec)

    # Save report
    report_text = "\n".join(report)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Report saved to: {REPORT_FILE}")
    print(f"{'='*80}")

    return report_text


if __name__ == "__main__":
    generate_report()
