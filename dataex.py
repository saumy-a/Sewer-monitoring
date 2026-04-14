import firebase_admin
from firebase_admin import credentials, db
import json
import csv

# Firebase setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sewer-3d62e-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Get data
ref = db.reference('sensor_data')
data = ref.get()

with open('firebase_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['temp','humidity','air','methane','distance','flow','status'])

    if data:
        for key, value in data.items():
            writer.writerow([
                value.get('temp'),
                value.get('humidity'),
                value.get('air'),
                value.get('methane'),
                value.get('distance'),
                value.get('flow'),
                value.get('status')
            ])

# Save to file
with open('firebase_data.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Data exported to firebase_data.json and firebase_data.csv")