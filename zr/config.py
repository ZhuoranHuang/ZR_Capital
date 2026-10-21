"""全局配置：从环境变量 / .env 读取。密钥只放 .env，不得硬编码。"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    longport_app_key: SecretStr | None = None
    longport_app_secret: SecretStr | None = None
    longport_access_token: SecretStr | None = None

    zr_data_root: Path = Path("./data_store")
    # 实盘总开关；实盘下单还需命令行 --live，见 CLAUDE.md
    zr_live: bool = False
    # 逗号分隔，例如 SPY.US,QQQ.US
    zr_symbols: str = "SPY.US,QQQ.US,MU.US"  # AMD 暂停（D-016）

    @property
    def symbols(self) -> list[str]:
        return [s.strip() for s in self.zr_symbols.split(",") if s.strip()]

    @property
    def has_longport_keys(self) -> bool:
        keys = (self.longport_app_key, self.longport_app_secret, self.longport_access_token)
        return all(k is not None and k.get_secret_value() for k in keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
