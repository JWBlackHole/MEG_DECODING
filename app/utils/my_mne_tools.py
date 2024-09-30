import mne
# res = mne.channels.get_builtin_montages()
# print(type(res))
# print(res)

res=mne.channels.read_custom_montage("/home/dataset/Data/gw_data/download/sub-01/ses-0/meg/sub-01_ses-0_acq-ELP_headshape.elp")
print(type(res))
print(res)
