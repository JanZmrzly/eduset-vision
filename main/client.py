import asyncio

from eduset.utils.logger import EdusetLogger
from eduset.utils.opcua import OPCUAClient


logger = EdusetLogger("client")


async def main():
    url = "opc.tcp://192.168.0.10:4840"
    api_endpoint = "http://localhost:8000/placement"

    control_signal = {"ns": "http://EdusetONE",
                      "i": 75}
    placement = {"ns": "http://EdusetONE",
                 "i": 19}

    client = OPCUAClient(url=url)

    try:
        await client.set_encryption() if client.encryption is True else None
        await client.run(control_signal, placement, time=2, api_endpoint=api_endpoint)
    except KeyboardInterrupt:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
