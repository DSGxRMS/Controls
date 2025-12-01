import csv 

# Helper function to skip header rows
def skip_header(reader, n):
    for _ in range(n):
        next(reader)
    return list(reader)

with open('./pathpoints.csv', 'r') as f:
    reader = csv.reader(f)
    # Ensure next(reader) is called to advance the iterator past the header
    next(reader) 
    # Convert remaining rows
    path = [[float(row[1]) + 15.0, -float(row[0])] for row in reader]

with open('./pathpoints_shifted.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(path)

print(path)
