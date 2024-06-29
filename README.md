## Data
### Downloading CPHDs
- Visit Capella Space open dataset and browse CPHDs here: https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-product-type/capella-open-data-cphd/collection.json
- Find a CPHD that is not too large < 2GB

With AWS CLI go to `Data file > Copy URL` and use the following command format to copy the CPHD. You will need to replace `http` with `s3` and remove `.s3.amazonaws.com`.
```bash
aws s3 cp s3://capella-open-data/data/2021/2/16/CAPELLA_C02_SM_CPHD_HH_20210216020741_20210216
020745/CAPELLA_C02_SM_CPHD_HH_20210216020741_20210216020745.cphd . --no-sign-request`
```
