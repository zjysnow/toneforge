import colour
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.prepared import prep

def calculate_gamut_metrics(
    target_primaries: np.ndarray, 
    reference_primaries: np.ndarray, 
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

    def safe_create_polygon(primaries: np.ndarray) -> Polygon | MultiPolygon:
        """Return a valid polygonal geometry, or an empty polygon."""
        primaries = np.asarray(primaries, dtype=float)
        if primaries.ndim != 2 or primaries.shape[1] != 2 or not np.isfinite(primaries).all():
            return Polygon()

        poly = Polygon(primaries)
        if not poly.is_valid:
            poly = make_valid(poly)

        if isinstance(poly, GeometryCollection):
            polygon_parts = [
                part
                for part in poly.geoms
                if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
            ]
            poly = unary_union(polygon_parts) if polygon_parts else Polygon()

        if not isinstance(poly, (Polygon, MultiPolygon)) or poly.is_empty:
            return Polygon()

        if not poly.is_valid:
            poly = poly.buffer(0)

        return poly if poly.is_valid else Polygon()
    
    # 转换为 numpy 数组保证数据类型安全
    target_primaries = np.asarray(target_primaries, dtype=float)
    reference_primaries = np.asarray(reference_primaries, dtype=float)


    if use_uv_space:
        target_primaries = xy_to_uv(target_primaries)
        reference_primaries = xy_to_uv(reference_primaries)

    target_polygon = safe_create_polygon(target_primaries)
    reference_polygon = safe_create_polygon(reference_primaries)

    area_target = target_polygon.area
    area_ref = reference_polygon.area

    with np.errstate(invalid='ignore'):
        intersection_shape = target_polygon.intersection(reference_polygon)
        union_shape = target_polygon.union(reference_polygon)

        intersection_area = intersection_shape.area
        union_area = union_shape.area

        return {
            "space": "CIE 1976 u'v'" if use_uv_space else "CIE 1931 xy",
            "coverage": (intersection_area / area_ref) * 100.0 if area_ref > 0 else 0.0,
        }


if __name__ == "__main__":
    from toneforge import color

    print(calculate_gamut_metrics([[0.6751, 0.3201],[0.2647, 0.6878],[0.1462, 0.0695]], color.DCI_P3, False))