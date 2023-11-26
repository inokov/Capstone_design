from faker import Faker
import csv

fake = Faker()

# Specify the column names
columns = ["Product ID", "Brand Name", "Product Name", "Ingredients", "Weight", "Expiring Date", "Certificate",
           "Country", "Price", "Location of Product in Store"]

# Generate 3000 rows of data
rows = []
for _ in range(3000):
    row = [
        fake.uuid4(),  # Product ID
        fake.company(),  # Brand Name
        fake.word(),  # Product Name
        fake.words(),  # Ingredients (list of words)
        fake.random_number(digits=3),  # Weight (in grams)
        fake.future_date(),  # Expiring Date
        fake.random_element(["FDA", "CE", "ISO", "HALAL", "KOSHER", "NON_GMO", "VEGAN FRIENDLY"]),  # Certificate
        fake.country(),  # Country
        fake.random_int(min=100, max=10000) / 100,  # Price (random integer divided by 100)
        fake.random_int(min=1, max=10)  # Location of Product in Store
    ]
    rows.append(row)

# Write the data to a CSV file
filename = "Database_of_Supermarket.csv"
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the header
    csvwriter.writerow(columns)

    # Write the rows
    csvwriter.writerows(rows)

print(f"Dump data has been generated and saved to {filename}")
