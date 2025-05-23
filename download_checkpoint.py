import gdown

FILE_ID='1wUIXP1fgpNq2QOWfqvJYwuMLoUpKYN8J'
FILE_NAME='lightning_logs.tar.gz'
URL=f"https://drive.google.com/uc?id={FILE_ID}"
gdown.download(f"{URL}", FILE_NAME)




