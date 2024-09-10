import mne
import os
import json
from pathlib import Path



def plot_sensor(root_path):
    """
    plot sensor topology
    """
    root_path = Path(root_path)
    mrk_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_task-0_markers.mrk'
    elp_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_acq-ELP_headshape.elp'
    hsp_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_acq-HSP_headshape.hsp'
    elp_pos_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_acq-ELP_headshape.pos'
    hsp_pos_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_acq-HSP_headshape.pos'
    coordsystem_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_coordsystem.json'
    con_path = root_path / 'sub-01/ses-0/meg/sub-01_ses-0_task-0_meg.con'

    # Read the marker file (fiducials)
    raw = mne.io.read_raw_kit(con_path, mrk=mrk_path, elp=elp_path, hsp=hsp_path, elp_pos=elp_pos_path, hsp_pos=hsp_pos_path, preload=False)

    with open(coordsystem_path, 'r') as f:
        coordsystem = json.load(f)

    # Plot the sensor topology
    fig = mne.viz.plot_alignment(raw.info, meg=('helmet', 'sensors'), coord_frame='meg')
    mne.viz.set_3d_title(figure=fig, title="KIT Sensor Topology")

    # Show the plot
    mne.viz.show()