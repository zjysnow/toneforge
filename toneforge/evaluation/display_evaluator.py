import colour
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.prepared import prep

from toneforge import color

def evaluate_display_color(
    target_primaries: np.ndarray, 
    reference_primaries: np.ndarray = color.DCI_P3, 
    use_uv_space: bool = False
):

    def xy_to_uv(xy: np.ndarray) -> np.ndarray:
        # 如果不用 colour，也可写成: return colour.xy_to_UCS_uv(xy)
        x, y = xy[:, 0], xy[:, 1]
        denom = -2 * x + 12 * y + 3
        # 加上小量 1e-10 避免除以 0
        denom = np.where(denom == 0, 1e-10, denom)
        u_prime = 4 * x / denom
        v_prime = 9 * y / denom
        return np.column_stack((u_prime, v_prime))

    target_primaries = np.asarray(target_primaries, dtype=np.float64)
    reference_primaries = np.asarray(reference_primaries, dtype=np.float64)

    if use_uv_space:
        target_primaries = xy_to_uv(target_primaries)
        reference_primaries = xy_to_uv(reference_primaries)

    target_polygon = Polygon(target_primaries)
    reference_polygon = Polygon(reference_primaries)

    area_target = target_polygon.area
    area_ref = reference_polygon.area

    with np.errstate(invalid='ignore'):
        intersection_shape = target_polygon.intersection(reference_polygon)
        union_shape = target_polygon.union(reference_polygon)

        intersection_area = intersection_shape.area
        union_area = union_shape.area

        return {
            "space": "CIE 1976 u'v'" if use_uv_space else "CIE 1931 xy",
            "coverage_rate": (intersection_area / area_ref) * 100.0 if area_ref > 0 else 0.0,
            "volume_ratio": (area_target / area_ref) * 100.0 if area_ref > 0 else 0.0,
            "iou": (intersection_area / union_area) * 100.0 if union_area > 0 else 0.0,
        }


if __name__ == "__main__":
    from toneforge import color

    print(evaluate_display_color([[0.6751, 0.3201],[0.2647, 0.6878],[0.1462, 0.0695]], color.DCI_P3, False))