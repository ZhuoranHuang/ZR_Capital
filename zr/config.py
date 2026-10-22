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
    longport_live_access_token: SecretStr | None = None
    longport_paper_access_token: SecretStr | None = None
    # 旧字段名，LONGPORT_LIVE_ACCESS_TOKEN 的兼容别名；见 longport_client.make_config 的改名提示
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
    def live_access_token(self) -> SecretStr | None:
        """实盘 token：优先 LONGPORT_LIVE_ACCESS_TOKEN，否则退回旧字段名
        LONGPORT_ACCESS_TOKEN（见 longport_client.make_config 的改名提示）。
        """
        return self.longport_live_access_token or self.longport_access_token

    @property
    def using_legacy_access_token_name(self) -> bool:
        """是否仍在用旧字段名 LONGPORT_ACCESS_TOKEN 而未设置 LONGPORT_LIVE_ACCESS_TOKEN。"""
        return self.longport_live_access_token is None and self.longport_access_token is not None

    def access_token_for(self, mode: str) -> SecretStr | None:
        if mode not in ("live", "paper"):
            raise ValueError(f"mode 必须是 'live' 或 'paper'：{mode!r}")
        return self.longport_paper_access_token if mode == "paper" else self.live_access_token

    def has_longport_keys(self, mode: str = "live") -> bool:
        token = self.access_token_for(mode)
        keys = (self.longport_app_key, self.longport_app_secret, token)
        return all(k is not None and k.get_secret_value() for k in keys)


@lru_cache
def get_settings() -> Settings:
    return Settings()
