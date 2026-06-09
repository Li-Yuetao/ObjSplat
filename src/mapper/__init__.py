from enum import Enum

class MapperState(Enum):
    MAPPING = 1
    ON_MAPPING = 2
    IDLE = 3
    
class GaussianColorType(Enum):
    Color = 'Color'
    Depth = 'Depth'
    Opacity = 'Opacity'
    Normal = 'Normal'
    Confidence = 'Confidence'
    Uncertainty = 'Uncertainty'
    Elipsoid = 'Elipsoid'

class MapperType(Enum):
    GSMap = 'GSMap'
    
def get_mapper(model_type:MapperType):
    if model_type == MapperType.GSMap:
        from mapper.gsmap import GSMap
        return GSMap
    else:
        raise ValueError(f"Model type {model_type} not supported")