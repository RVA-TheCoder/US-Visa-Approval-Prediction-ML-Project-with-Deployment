# Code 1 : For checking if the system environment variable is set properly or not.
# Restart the system after adding the system environment variable and then run below code.

# import os
# #from us_visa.constants import MONGODB_URL_KEY
# #print(os.getenv(MONGODB_URL_KEY))

# from us_visa.constants import MONGODB_URI_KEY
# print(MONGODB_URI_KEY)

from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.constants import SCHEMA_FILEPATH
data_schema = read_yaml_file(filepath=SCHEMA_FILEPATH)
print(data_schema)





