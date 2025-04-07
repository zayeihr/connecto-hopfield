#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify access to required data files with the updated configuration.
This script checks if the critical data files are accessible and correctly formatted.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
import glob
from datetime import datetime

# Add src directory to Python path to allow importing utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)

def setup_logging():
    """Set up logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def check_regional_timeseries(base_dir, subject_id="sub-01", run_id="01"):
    """Check if regional timeseries files are accessible."""
    regional_ts_dir = os.path.join(base_dir, "regional-timeseries")
    
    if not os.path.exists(regional_ts_dir):
        logging.error(f"Regional timeseries directory not found: {regional_ts_dir}")
        return False
    
    logging.info(f"Regional timeseries directory exists: {regional_ts_dir}")
    
    # Check for a specific subject/run or just list the first few files
    ts_pattern = os.path.join(regional_ts_dir, f"{subject_id}_task-pain_run-{run_id}_timeseries.tsv")
    ts_files = glob.glob(ts_pattern)
    
    if not ts_files:
        # Try with a more general pattern
        ts_pattern = os.path.join(regional_ts_dir, "*_timeseries.tsv")
        ts_files = glob.glob(ts_pattern)
        if not ts_files:
            logging.error(f"No timeseries files found in {regional_ts_dir}")
            return False
    
    # List some of the found files
    logging.info(f"Found {len(ts_files)} timeseries files. Examples:")
    for i, ts_file in enumerate(ts_files[:5]):
        logging.info(f"  {i+1}. {os.path.basename(ts_file)}")
    
    # Try to read one of the files to check format
    try:
        test_file = ts_files[0]
        logging.info(f"Testing file read: {test_file}")
        ts_data = pd.read_csv(test_file, sep='\t', header=None)
        logging.info(f"  Successfully read data with shape: {ts_data.shape}")
        return True
    except Exception as e:
        logging.error(f"Error reading timeseries file {test_file}: {e}")
        return False

def check_fd_files(base_dir, subject_id="sub-01", run_id="01"):
    """Check if framewise displacement files are accessible."""
    fd_dir = os.path.join(base_dir, "func_preproc", "mc_fd")
    
    if not os.path.exists(fd_dir):
        logging.error(f"FD directory not found: {fd_dir}")
        return False
    
    logging.info(f"FD directory exists: {fd_dir}")
    
    # Check for FD files
    fd_pattern = os.path.join(fd_dir, f"{subject_id}_task-pain_run-{run_id}*")
    fd_files = glob.glob(fd_pattern)
    
    if not fd_files:
        # Try with a more general pattern
        fd_pattern = os.path.join(fd_dir, "*")
        fd_files = glob.glob(fd_pattern)
        if not fd_files:
            logging.error(f"No FD files found in {fd_dir}")
            return False
    
    # List some of the found files
    logging.info(f"Found {len(fd_files)} FD files. Examples:")
    for i, fd_file in enumerate(fd_files[:5]):
        logging.info(f"  {i+1}. {os.path.basename(fd_file)}")
    
    # Try to read one of the files to check format
    try:
        test_file = fd_files[0]
        logging.info(f"Testing file read: {test_file}")
        with open(test_file, 'r') as f:
            fd_values = [float(line.strip()) for line in f if line.strip()]
        logging.info(f"  Successfully read data with length: {len(fd_values)}")
        return True
    except Exception as e:
        logging.error(f"Error reading FD file {test_file}: {e}")
        return False

def check_qc_summary_files(base_dir):
    """Check if QC summary files are accessible."""
    qc_summary_dir = os.path.join(base_dir, "func_preproc")
    
    # Expected files
    fd_max_file = os.path.join(qc_summary_dir, "popFD_max.txt")
    percent_scrubbed_file = os.path.join(qc_summary_dir, "pop_percent_scrubbed.txt")
    
    success = True
    
    # Check FD max file
    if os.path.exists(fd_max_file):
        logging.info(f"FD max file exists: {fd_max_file}")
        try:
            fd_data = pd.read_csv(fd_max_file, sep='\s+', header=None)
            logging.info(f"  Successfully read FD max data with shape: {fd_data.shape}")
        except Exception as e:
            logging.error(f"Error reading FD max file: {e}")
            success = False
    else:
        logging.error(f"FD max file not found: {fd_max_file}")
        success = False
    
    # Check percent scrubbed file
    if os.path.exists(percent_scrubbed_file):
        logging.info(f"Percent scrubbed file exists: {percent_scrubbed_file}")
        try:
            scrub_data = pd.read_csv(percent_scrubbed_file, sep='\s+', header=None)
            logging.info(f"  Successfully read percent scrubbed data with shape: {scrub_data.shape}")
        except Exception as e:
            logging.error(f"Error reading percent scrubbed file: {e}")
            success = False
    else:
        logging.error(f"Percent scrubbed file not found: {percent_scrubbed_file}")
        success = False
    
    return success

def check_events_files(base_dir, subject_id="sub-01", run_id="01"):
    """Check if events.tsv files are accessible."""
    # Events files are expected to be in BIDS format
    events_file = os.path.join(base_dir, subject_id, "func", f"{subject_id}_task-pain_run-{run_id}_events.tsv")
    
    if os.path.exists(events_file):
        logging.info(f"Events file exists: {events_file}")
        try:
            events_data = pd.read_csv(events_file, sep='\t')
            logging.info(f"  Successfully read events data with shape: {events_data.shape}")
            logging.info(f"  Columns: {events_data.columns.tolist()}")
            return True
        except Exception as e:
            logging.error(f"Error reading events file: {e}")
            return False
    else:
        logging.error(f"Events file not found: {events_file}")
        
        # Try to find events files with a more general pattern
        events_pattern = os.path.join(base_dir, "**", "*_events.tsv")
        events_files = glob.glob(events_pattern, recursive=True)
        
        if events_files:
            logging.info(f"Found {len(events_files)} possible events files. Examples:")
            for i, ef in enumerate(events_files[:5]):
                logging.info(f"  {i+1}. {ef}")
        else:
            logging.error("No events.tsv files found in the entire directory structure.")
        
        return False

def main(config_path):
    """Main function to test data access."""
    setup_logging()
    logging.info(f"Testing data access with config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get base directory
    base_dir = config['paths']['base_dir']
    logging.info(f"Base directory: {base_dir}")
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        logging.error(f"Base directory not found: {base_dir}")
        sys.exit(1)
    
    # Check component directories and files
    success = True
    
    # Check regional timeseries
    logging.info("\n--- Testing Regional Timeseries Access ---")
    ts_success = check_regional_timeseries(base_dir)
    success = success and ts_success
    
    # Check FD files
    logging.info("\n--- Testing Framewise Displacement Files Access ---")
    fd_success = check_fd_files(base_dir)
    success = success and fd_success
    
    # Check QC summary files
    logging.info("\n--- Testing QC Summary Files Access ---")
    qc_success = check_qc_summary_files(base_dir)
    success = success and qc_success
    
    # Check events files
    logging.info("\n--- Testing Events Files Access ---")
    events_success = check_events_files(base_dir)
    success = success and events_success
    
    # Summary
    logging.info("\n--- Summary ---")
    if success:
        logging.info("✅ All data access tests PASSED. The data structure appears compatible with the configuration.")
    else:
        logging.warning("❌ Some data access tests FAILED. Review the log for details and adjust configuration as needed.")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data access with the ACHNN configuration.")
    parser.add_argument("config", help="Path to the configuration YAML file (e.g., configs/achnn_config.yaml)")
    args = parser.parse_args()
    
    main(args.config) 