import communication

communicator = communication.Communication()

sample_string = "this is a test sentnce for the natural language processing model"

for letter in sample_string:
    print(f"Testing: {letter}.")
    print(letter)
    communicator.new_letter(letter)
