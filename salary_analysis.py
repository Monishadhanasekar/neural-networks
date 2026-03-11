import csv
from collections import defaultdict

def calculate_average_salaries():
    department_data = defaultdict(list)
    
    # Read the CSV file
    with open('employees.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            department = row['department']
            salary = float(row['salary'])
            department_data[department].append(salary)
    
    # Calculate averages
    averages = {}
    for department, salaries in department_data.items():
        averages[department] = sum(salaries) / len(salaries)
    
    # Print formatted report
    print("Department Salary Report")
    print("=" * 30)
    for department, avg_salary in sorted(averages.items()):
        print(f"{department}: ${avg_salary:,.2f}")
    
    return averages

if __name__ == "__main__":
    calculate_average_salaries()