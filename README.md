# SAR Toolkit
A toolkit for working with Syntehtic Aperture Radar Images (SAR)

## Data
A quick example of how to get open-source SAR data

### Downloading NITFs and CPHDs
__NOTE:__ For NITFs use the GEOTIFF; this is the SIDD and is what's shown in the preview; the SICD is super slanty and looks bad; you'll need to load the tiff with `tifffile` and then remap it with the regular sarpy `Desnity` remap

- Visit Capella Space open dataset and browse CPHDs here: https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-product-type/capella-open-data-cphd/collection.json
- Find a CPHD that is not too large < 2GB

With AWS CLI go to `Data file > Copy URL` and use the following command format to copy the CPHD. You will need to replace `http` with `s3` and remove `.s3.amazonaws.com`.
```bash
aws s3 cp --no-sign-requests3://capella-open-data/data/2021/2/16/CAPELLA_C02_SM_CPHD_HH_20210216020741_20210216
020745/CAPELLA_C02_SM_CPHD_HH_20210216020741_20210216020745.cphd .
```

If the file cannot be found, you can just manually download it with the `Download` button.

## Usage

### Chipping a full-scene SAR Image
```bash
python scripts/chip_sicd.py /mnt/d/datasets/capella-sar/sicds/CAPELLA_C13_SP_SICD_HH_20241127120857_20241127120930.ntf --chip-size 512 --remap
```
