import asyncio
import socket
import warnings
import random
import requests

from asyncua import Client, ua, Node
from asyncua.crypto.cert_gen import setup_self_signed_certificate
from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256
from asyncua.crypto.validator import CertificateValidator, CertificateValidatorOptions
from cryptography.x509.oid import ExtendedKeyUsageOID
from cryptography.utils import CryptographyDeprecationWarning
from pathlib import Path

warnings.filterwarnings(action="ignore", category=CryptographyDeprecationWarning)


class SubHandler:
    def datachange_notification(self, node: Node, val, data):
        pass

    def event_notification(self, event: ua.EventNotificationList):
        pass

    def status_change_notification(self, status: ua.StatusChangeNotification):
        pass


class OPCUAClient(Client):
    def __init__(self, url: str, encryption=True) -> None:
        super(OPCUAClient, self).__init__(
            url=url
        )

        self.encryption = encryption
        self.url = url
        self.secure_channel_timeout = 30000
        self.session_timeout = 30000
        self.placement_node: Node | None = None
        self.buffer: dict = {}

        self.name = "EdusetONE Vision OPC UA Client"
        self.description = "EdusetONE Vision OPC UA Client"
        self.certificate_path = "../eduset/utils/peer-certificate-example-3.der"
        self.private_key_path = "../eduset/utils/peer-private-key-example-3.pem"

    async def set_placement_node(self, ns: str | int, i: int) -> None:
        nsidx = await self.get_namespace_index(ns) if type(ns) is str else ns

        self.placement_node = self.get_node(f"ns={nsidx}; i={i}")

    async def set_encryption(self) -> None:
        host_name = socket.gethostname()
        client_app_uri = f"urn:{host_name}:foobar:myselfsignedclient"

        await setup_self_signed_certificate(Path(self.private_key_path).resolve(),
                                            Path(self.certificate_path).resolve(),
                                            client_app_uri,
                                            host_name,
                                            cert_use=[ExtendedKeyUsageOID.CLIENT_AUTH],
                                            subject_attrs={
                                                'countryName': 'CZ',
                                                'stateOrProvinceName': 'JMK',
                                                'localityName': 'Brno',
                                                'organizationName': "Zmrzly",
                                            })

        self.application_uri = client_app_uri

        await self.set_security(
            SecurityPolicyBasic256Sha256,
            certificate=self.certificate_path,
            private_key=self.private_key_path,
        )

        validator = CertificateValidator(CertificateValidatorOptions.EXT_VALIDATION |
                                         CertificateValidatorOptions.PEER_SERVER)

        self.certificate_validator = validator

    async def run(self, ns: str | int, i: int, time: int, api_url: str) -> None:
        handler = SubHandler()

        while True:
            try:
                async with self:
                    print(f"Connected to server {self.url}")
                    await self.set_placement_node(ns, i)

                    subscription = await self.create_subscription(period=500, handler=handler)
                    nodes = [
                        self.placement_node,
                        self.get_node(ua.ObjectIds.Server_ServerStatus_CurrentTime),
                    ]
                    await subscription.subscribe_data_change(nodes)

                    await self.get_buffer()

                    while True:
                        await asyncio.sleep(0.01)
                        await self.check_connection()
                        placement = get_placement_from_api(api_url)
                        await self.rewrite_buffer(placement)

            except (ConnectionError, ua.UaError) as e:
                print(e)
                print(f"No server {self.url}. Reconnecting in {time} seconds..")
                await asyncio.sleep(time)
            except OSError as e:
                print(e)
                print(f"No server {self.url}. Reconnecting in {time} seconds..")
                await asyncio.sleep(time)

    async def get_buffer(self) -> None:
        if not None:
            children = await self.placement_node.get_children()
            for i, child in enumerate(children):
                grandchildren = await child.get_children()

                temp_dict = {}
                for grandchild in grandchildren:
                    name = await grandchild.read_browse_name()
                    temp_dict[str(name.Name)] = grandchild

                self.buffer[str(i)] = temp_dict

    async def rewrite_buffer(self, placement: dict | None) -> None:
        if placement:
            shorter_dict = placement if len(placement) < len(self.buffer) else self.buffer

            for key, item in shorter_dict.items():
                placement_item = placement[key]
                buffer_item = self.buffer[key]

                new_name = ua.DataValue(ua.Variant(placement_item["name"], ua.VariantType.String))
                new_x = ua.DataValue(ua.Variant(placement_item["x"], ua.VariantType.UInt16))
                new_y = ua.DataValue(ua.Variant(placement_item["y"], ua.VariantType.UInt16))
                new_angle = ua.DataValue(ua.Variant(placement_item["angle"], ua.VariantType.Float))

                await buffer_item["name"].write_value(new_name)
                await buffer_item["x"].write_value(new_x)
                await buffer_item["y"].write_value(new_y)
                await buffer_item["angle"].write_value(new_angle)


def get_placement() -> dict:
    placement = {
                    "0": {
                        "name": "triangle",
                        "x": random.randint(a=0, b=2448),
                        "y": random.randint(a=0, b=2448),
                        "angle": random.uniform(a=-3, b=3),
                    },
                    "1": {
                        "name": "circle",
                        "x": random.randint(a=0, b=2448),
                        "y": random.randint(a=0, b=2448),
                        "angle": random.uniform(a=-3, b=3),
                    }
                }
    return placement


def get_placement_from_api(api_url: str) -> dict | None:
    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error while collecting data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


async def main() -> None:
    url = "opc.tcp://192.168.0.10:4840"
    ns = "http://EdusetONE"
    i = 19

    api_url = "http://localhost:8000/placement"
    client = OPCUAClient(url=url)

    try:
        await client.set_encryption() if client.encryption is True else None
        await client.run(ns, i, time=2, api_url=api_url)
    except KeyboardInterrupt:
        await client.disconnect()
