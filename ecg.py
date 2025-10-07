'''
import os
import re
from utils import *
import pandas as pd
import heartpy as hp
from ecgdetectors import Detectors
from QRSDetectorOffline import QRSDetectorOffline

def ecg_preprocessor(detector):
    """
    Detects the QRS complexes in the given ECG signal and extracts the HRV features for every patient
    Parameters: detector (str): The detector to use.
    Returns: None
    """    
    files = [i for i in os.listdir() if(re.search("VP*", i))]
    final_df_list = []
    for i in range(0,len(files)):    
        ecg_df = pd.read_csv(os.path.join(files[i], os.listdir(files[i])[1]), delimiter='\t', names=["ecg", "time", "raw"], header=None, index_col=False)
        ecg_df = ecg_df.drop(columns = "raw")
        ecg_df = ecg_df.sort_values(['time'])
        ecg_df = ecg_df[["time", "ecg"]]
        ecg_df.reset_index(drop=True, inplace = True)
        ecg_df.to_csv(path_or_buf = os.path.join(files[i], "ecg.csv"),index = False)

        sca_ecg = hp.scale_data(ecg_df["ecg"], lower = 0, upper = 3)

        exposure_period_df = pd.read_csv(os.path.join(files[i], os.listdir(files[i])[-1]), delimiter='\t', names=["event", "s_time", "e_time"], header=None, index_col=False)
        
        if (len(exposure_period_df.loc[(exposure_period_df["event"]=="BIOFEEDBACK-OXYGEN-TRAININGS") | (exposure_period_df["event"] == "BIOFEEDBACK-REST")]) == 2):
            exposure_period_df = exposure_period_df.loc[exposure_period_df["event"]!="BIOFEEDBACK-OXYGEN-TRAININGS"].copy()
            exposure_period_df = exposure_period_df.reset_index()

        if(detector == "pan-tompkins"):
            qrs_detector = QRSDetectorOffline(ecg_data_path=os.path.join(files[i],"ecg.csv"), verbose=True, log_data=True, plot_data=False, show_plot=False)
            peaks = extract_peaks()
        elif(detector == "hamilton"):
            detectors = Detectors(100)
            r_peaks = detectors.hamilton_detector(sca_ecg)
            peaks = ecg_df.iloc[r_peaks]

        peaks = peaks.rename(columns = {"ecg" : "ecg_measurement", "time" : "timestamp"})
        peaks.drop_duplicates(subset = ['timestamp'], keep = 'first', inplace = True)
        peaks = extract_hr(peaks)
        peaks = extract_NNI(peaks)
        final_df = adv_features(peaks, exposure_period_df)
        subject_no = files[i]
        final_df.insert(0, "subject", subject_no)
        final_df_list.append(final_df)

    ecg_df = pd.concat(final_df_list)
    ecg_df = ecg_df[(ecg_df["sdNN"].isnull() == False) & (ecg_df["RMSSD"].isnull() == False)]
    _ = ecg_df.groupby(['subject', 'event'])["mean_hr"].agg(['mean'])
    g = _.groupby(['event'])["mean"].agg(['mean'])

    high_clips = g.nlargest(7, 'mean').index[1:4].tolist()
    medium_clips = g.nlargest(7, 'mean').index[4:].tolist()
    high_df = ecg_df.loc[(ecg_df["event"].isin(high_clips))]
    medium_df = ecg_df.loc[(ecg_df["event"].isin(medium_clips))]
    high_df["anxiety"] = 3
    medium_df["anxiety"] = 2
    bio_df = ecg_df[(ecg_df["event"] == "BIOFEEDBACK-REST") | (ecg_df["event"] == "BIOFEEDBACK-OXYGEN-TRAININGS")]
    low_df = bio_df.groupby(['subject']).tail(18)
    low_df["anxiety"] = 1

    ecg_preprocessed = pd.concat([high_df, medium_df, low_df])
    ecg_preprocessed.to_csv(path_or_buf = "ecg_processed.csv", index=False)
'''

import os
import re
import csv
import pandas as pd
import heartpy as hp
from utils import *
from ecgdetectors import Detectors
from QRSDetectorOffline import QRSDetectorOffline


def detect_delimiter(path):
    """Auto-detect CSV or TSV delimiter."""
    with open(path, 'r') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except Exception:
            return ','  # fallback


def ecg_preprocessor(detector):
    """
    Detects the QRS complexes in the given ECG signal and extracts HRV features for every patient.
    """
    files = [i for i in os.listdir() if re.search("VP*", i)]
    final_df_list = []

    for i in range(len(files)):
        # Locate the ECG file
        ecg_file_path = os.path.join(files[i], os.listdir(files[i])[1])
        print(f"📂 Processing {ecg_file_path} in chunks...")

        # Detect the delimiter dynamically
        delimiter = detect_delimiter(ecg_file_path)
        print(f"🧭 Detected delimiter for {files[i]}: '{delimiter}'")

        # Read the ECG file in chunks
        chunks = pd.read_csv(
            ecg_file_path,
            delimiter=delimiter,
            names=["time", "ecg", "raw"],
            header=None,
            index_col=False,
            chunksize=500000
        )

        # Combine all chunks
        ecg_df_list = []
        for chunk in chunks:
            # Keep only time and ecg columns
            if "time" in chunk.columns and "ecg" in chunk.columns:
                ecg_df_list.append(chunk[["time", "ecg"]])
        ecg_df = pd.concat(ecg_df_list, ignore_index=True)

        # Log early rows
        print(f"🔍 Raw sample for {files[i]}:\n", ecg_df.head(5))

        # Ensure numeric conversion
        ecg_df["time"] = pd.to_numeric(ecg_df["time"], errors="coerce")
        ecg_df["ecg"] = pd.to_numeric(ecg_df["ecg"], errors="coerce")
        ecg_df = ecg_df.dropna(subset=["time", "ecg"])

        # If still empty, skip
        if ecg_df.empty:
            print(f"⚠️ Skipping {files[i]} — ECG file had no numeric data.")
            continue

        # Sort and save smaller CSV
        ecg_df = ecg_df.sort_values(["time"])
        ecg_df.reset_index(drop=True, inplace=True)
        small_csv_path = os.path.join(files[i], "ecg.csv")
        ecg_df.to_csv(small_csv_path, index=False)
        print(f"✅ Saved smaller ECG CSV at {small_csv_path}")
        print(f"🔍 Sample ECG values for {files[i]}:\n", ecg_df.head(5))

        # Scale data
        try:
            sca_ecg = hp.scale_data(ecg_df["ecg"], lower=0, upper=3)
        except Exception as e:
            print(f"⚠️ Scaling failed for {files[i]}: {e}")
            continue

        # Read exposure events
        exposure_path = os.path.join(files[i], os.listdir(files[i])[-1])
        delimiter_exp = detect_delimiter(exposure_path)
        exposure_period_df = pd.read_csv(
            exposure_path,
            delimiter=delimiter_exp,
            names=["event", "s_time", "e_time"],
            header=None,
            index_col=False,
        )

        # Handle duplicate event issue
        if len(exposure_period_df.loc[
            (exposure_period_df["event"] == "BIOFEEDBACK-OXYGEN-TRAININGS")
            | (exposure_period_df["event"] == "BIOFEEDBACK-REST")
        ]) == 2:
            exposure_period_df = exposure_period_df.loc[
                exposure_period_df["event"] != "BIOFEEDBACK-OXYGEN-TRAININGS"
            ].copy()
            exposure_period_df.reset_index(drop=True, inplace=True)

        # QRS detection
        try:
            if detector == "pan-tompkins":
                qrs_detector = QRSDetectorOffline(
                    ecg_data_path=small_csv_path,
                    verbose=True,
                    log_data=True,
                    plot_data=False,
                    show_plot=False,
                )
                peaks = extract_peaks()
            elif detector == "hamilton":
                detectors = Detectors(100)
                r_peaks = detectors.hamilton_detector(sca_ecg)
                peaks = ecg_df.iloc[r_peaks]
            else:
                print(f"⚠️ Unknown detector: {detector}, skipping {files[i]}")
                continue
        except Exception as e:
            print(f"❌ QRS detection failed for {files[i]}: {e}")
            continue

        # Feature extraction
        peaks = peaks.rename(columns={"ecg": "ecg_measurement", "time": "timestamp"})
        peaks.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
        peaks = extract_hr(peaks)
        peaks = extract_NNI(peaks)
        final_df = adv_features(peaks, exposure_period_df)
        final_df.insert(0, "subject", files[i])

        required_columns = ['sdNN', 'RMSSD']
        if any(col not in final_df.columns for col in required_columns):
            print(f"Skipping {files[i]}: missing essential columns {required_columns}")
            continue

        final_df_list.append(final_df)

    # Combine all results
    if not final_df_list:
        print("❌ No valid ECG data processed — check your files’ formatting.")
        return

    ecg_df = pd.concat(final_df_list)
    ecg_df = ecg_df[(ecg_df["sdNN"].notnull()) & (ecg_df["RMSSD"].notnull())]
    _ = ecg_df.groupby(['subject', 'event'])["mean_hr"].agg(['mean'])
    g = _.groupby(['event'])["mean"].agg(['mean'])

    high_clips = g.nlargest(7, 'mean').index[1:4].tolist()
    medium_clips = g.nlargest(7, 'mean').index[4:].tolist()
    high_df = ecg_df.loc[ecg_df["event"].isin(high_clips)]
    medium_df = ecg_df.loc[ecg_df["event"].isin(medium_clips)]
    high_df["anxiety"] = 3
    medium_df["anxiety"] = 2

    bio_df = ecg_df[
        (ecg_df["event"] == "BIOFEEDBACK-REST")
        | (ecg_df["event"] == "BIOFEEDBACK-OXYGEN-TRAININGS")
    ]
    low_df = bio_df.groupby(['subject']).tail(18)
    low_df["anxiety"] = 1

    ecg_preprocessed = pd.concat([high_df, medium_df, low_df])
    ecg_preprocessed.to_csv("ecg_processed.csv", index=False)
    print("🎉 Done! Saved as ecg_processed.csv")
