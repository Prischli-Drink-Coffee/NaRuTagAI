import os
import asyncio
from contextlib import asynccontextmanager
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

CHUNK_SIZE = 10 * 1024 * 1024  # Размер порции в байтах (10 МБ)

class S3Client:
    def __init__(self, access_key: str, secret_key: str, endpoint_url: str, bucket_name: str):
        self.config = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "endpoint_url": endpoint_url,
        }
        self.bucket_name = bucket_name
        self.session = get_session()

    @asynccontextmanager
    async def get_client(self):
        async with self.session.create_client("s3", **self.config) as client:
            yield client

    async def upload_file(self, file_path: str, object_name: str):
        try:
            async with self.get_client() as client:
                with open(file_path, "rb") as file:  # Открываем файл в бинарном режиме
                    await client.put_object(
                        Bucket=self.bucket_name,
                        Key=object_name,
                        Body=file,
                    )
                print(f"File {object_name} uploaded to {self.bucket_name}")
        except ClientError as e:
            print(f"Error uploading file: {e}")

    async def upload_file_in_chunks(self, file_path: str):
        part_number = 0
        try:
            with open(file_path, "rb") as file:
                while True:
                    chunk = file.read(CHUNK_SIZE)
                    if not chunk:  # Если нет данных, выходим из цикла
                        break
                    part_number += 1
                    object_name = f"{os.path.basename(file_path)}.part{part_number}"
                    await self.upload_chunk(chunk, object_name)  # Загружаем часть
        except Exception as e:
            print(f"Error uploading file in chunks: {e}")

    async def upload_chunk(self, chunk: bytes, object_name: str):
        try:
            async with self.get_client() as client:
                await client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=chunk,
                )
                print(f"Chunk {object_name} uploaded to {self.bucket_name}")
        except ClientError as e:
            print(f"Error uploading chunk: {e}")

    async def download_file_in_chunks(self, base_name: str, parts: int, destination_path: str):
        with open(destination_path, "wb") as outfile:
            for part_number in range(1, parts + 1):
                object_name = f"{base_name}.part{part_number}"
                try:
                    async with self.get_client() as client:
                        response = await client.get_object(Bucket=self.bucket_name, Key=object_name)
                        data = await response["Body"].read()
                        outfile.write(data)
                        print(f"Downloaded {object_name} to {destination_path}")
                except ClientError as e:
                    print(f"Error downloading part {part_number}: {e}")

async def main():
    s3_client = S3Client(
        access_key="6a80caeb53cb433da21014b2da5fb30d",
        secret_key="33799c6dcd35450fa4df11fdaf09b8a4",
        endpoint_url="https://s3.ru-1.storage.selcloud.ru",
        bucket_name="6554674",
    )

    # Загрузка .zip файла в части
    await s3_client.upload_file_in_chunks("C:/App/ReactProject/domains/NaRuTagAI/server/data/audio.zip")

    # Скачивание частей и сборка в единый файл
    # await s3_client.download_file_in_chunks("audio.zip", part_number, "downloaded_audio.zip")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
