# src/config.py

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # REMOVENDO DEFAULTS INSEGUROS: Pydantic exige que estas sejam definidas no ambiente.
    DB_HOST: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    
    DB_PORT: int = 5432
    TZ: str = "America/Sao_Paulo"

    @property
    def database_url(self) -> str:
        # Retorna a URL *crua* (postgresql://), a substituição do driver é feita no db.py
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

settings = Settings()
