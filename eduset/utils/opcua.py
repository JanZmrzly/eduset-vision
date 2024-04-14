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

from eduset.utils.logger import EdusetLogger

warnings.filterwarnings(action="ignore", category=CryptographyDeprecationWarning)


logger = EdusetLogger(__name__)


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
        self.control_signal_node: Node | None = None
        self.buffer: dict = {}

        self.name = "EdusetONE Vision OPC UA Client"
        self.description = "EdusetONE Vision OPC UA Client"
        self.certificate_path = "../eduset/utils/peer-certificate-example-3.der"
        self.private_key_path = "../eduset/utils/peer-private-key-example-3.pem"

    async def set_placement_node(self, ns: str | int, i: int) -> None:
        nsidx = await self.get_namespace_index(ns) if type(ns) is str else ns

        self.placement_node = self.get_node(f"ns={nsidx}; i={i}")

    async def set_control_signal_node(self, ns: str | int, i: int) -> None:
        nsidx = await self.get_namespace_index(ns) if type(ns) is str else ns

        self.control_signal_node = self.get_node(f"ns={nsidx}; i={i}")

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

    async def run(self, control_signal: dict, placement: dict, time: int, api_endpoint: str) -> None:
        handler = SubHandler(self, api_endpoint)

        while True:
            try:
                async with self:
                    logger.info(f"Connected to server {self.url}")
                    await self.set_control_signal_node(control_signal["ns"], control_signal["i"])
                    await self.set_placement_node(placement["ns"], placement["i"])

                    subscription = await self.create_subscription(period=300, handler=handler)
                    nodes = [
                        self.control_signal_node,
                    ]
                    await subscription.subscribe_data_change(nodes)
                    await self.get_buffer()

                    while True:
                        await asyncio.sleep(1)
                        await self.check_connection()

            except (ConnectionError, ua.UaError) as e:
                logger.error(e)
                logger.error(f"No server {self.url}. Reconnecting in {time} seconds..")
                await asyncio.sleep(time)
            except OSError as e:
                logger.error(e)
                logger.error(f"No server {self.url}. Reconnecting in {time} seconds..")
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

            for key, item in placement.items():
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

    async def fill_placement(self, placement: dict) -> dict:
        for i in range(len(placement), len(self.buffer)):
            placement[f"{i}"] = {'name': '', 'x': 0, 'y': 0, 'angle': 0.0}

        return placement


class SubHandler:
    def __init__(self, client: OPCUAClient, api_endpoint: str):
        self.client = client
        self.api_endpoint = api_endpoint

    async def datachange_notification(self, node: Node, val, data):
        if val:
            placement = get_placement_from_api(self.api_endpoint)
            placement = await self.client.fill_placement(placement)
            await self.client.rewrite_buffer(placement)

    def event_notification(self, event: ua.EventNotificationList):
        pass

    def status_change_notification(self, status: ua.StatusChangeNotification):
        pass


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
            logger.error(f"Error while collecting data: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
