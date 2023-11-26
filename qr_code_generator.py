import qrcode
import os

# Function to generate a QR code for the given data
def generate_qr(data, file_path):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(file_path)

# Input the word for which you want to generate a QR code
word = input("Enter the word for the QR code: ")

# Specify the file path for the QR code image
file_path = os.path.expanduser("~/Desktop/Univer/qr_generator/Ready_QR/mmmmmm.png")

# Generate the QR code
generate_qr(word, file_path)

print(f"QR code for '{word}' generated successfully. Saved as '{file_path}'.")
