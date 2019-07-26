
import adal
from pypowerbi.dataset import Column, Table, Dataset
from pypowerbi.client import PowerBIClient


# you might need to change these, but i doubt it
authority_url = 'https://login.windows.net/common'
resource_url = 'https://analysis.windows.net/powerbi/api'
api_url = 'https://api.powerbi.com'

redirectUri = 'https://login.live.com/oauth20_desktop.srf'
authorityUri = 'https://login.windows.net/common/oauth2/authorize'

# change these to your credentials
client_id = 'acbd70d9-cf76-427c-9466-c8d4a841221c'
username = 'andreas.schroeder@arpeggioqi.com'
password = ''

# first you need to authenticate using adal
context = adal.AuthenticationContext(authority=authority_url,
                                     validate_authority=True,
                                     api_version=None)

context

# get your authentication token
token = context.acquire_token_with_username_password(resource=resource_url,
                                                     client_id=client_id,
                                                     username=username,
                                                     password=password)

user_code_info = context.acquire_user_code(resource_url, client_id, language=None)
print(user_code_info)
token = context.acquire_token_with_device_code(resource_url, user_code_info, client_id)

print(token)



# create your powerbi api client
client = PowerBIClient(api_url, token)

# create your columns
columns = []
columns.append(Column(name='id', data_type='Int64'))
columns.append(Column(name='name', data_type='string'))
columns.append(Column(name='is_interesting', data_type='boolean'))
columns.append(Column(name='cost_usd', data_type='double'))
columns.append(Column(name='purchase_date', data_type='datetime'))

# create your tables
tables = []
tables.append(Table(name='AnExampleTableName', columns=columns))

# create your dataset
dataset = Dataset(name='AnExampleDatasetName', tables=tables)

# post your dataset!
ds = client.datasets.post_dataset(dataset)



# explore
dss = client.datasets.get_datasets()
[ds.name for ds in dss]
ds = dss[0]

ts = client.datasets.get_tables(ds.id)
[t.name for t in ts]
t = ts[0]

# add rows
from pypowerbi.dataset import Row, RowEncoder
import datetime

r = {'id': 1112,
     'name': 'test',
     'is_interesting': False,
     'cost_usd': 132.3,
     'purchase_date': '2018-01-01'
     }
print(r)
r = Row(**r)
print(r)

client.datasets.post_rows(ds.id, 'AnExampleTableName', [r])

client.datasets.delete_rows(ds.id, 'AnExampleTableName')

rows = [r]
row_encoder = RowEncoder()
json_dict = {
    'rows': [row_encoder.default(x) for x in rows]
}

