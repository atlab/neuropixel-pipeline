from pydantic import BaseModel, conint


class ScanKey(BaseModel, from_attributes=True):
    animal_id: conint(ge=0, le=2_147_483_647)
    session: conint(ge=0, le=32_767)
    scan_idx: conint(ge=0, le=32_767)
