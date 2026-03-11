from prime import is_prime

# Test numbers
test_numbers = [1, 2, 13, 15, 97, 100]

print("Testing is_prime function:")
for num in test_numbers:
    result = is_prime(num)
    print(f"is_prime({num}) = {result}")