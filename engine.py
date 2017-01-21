import json
from typing import NamedTuple, List, TextIO, Tuple, Iterator

import math

import pygame

SIZE = WIDTH, HEIGHT = 800, 600
BACKGROUND = (0.1875, 0.125, 0.0625, 1.0)
UNIT_SCALE = 80.0


class Vertex(NamedTuple):
    # coordinates
    x: float
    y: float
    z: float
    # color
    r: float
    g: float
    b: float
    a: float
    # normal vector for lighting
    nx: float
    ny: float
    nz: float


class Triangle(NamedTuple):
    vertices: List[Vertex]


class Model:
    def __init__(self,
                 triangles: List[Triangle],
                 x: float, y: float, z: float,
                 scale_x: float, scale_y: float, scale_z: float,
                 rotate_x: float, rotate_y: float, rotate_z: float,
                 origin_x: float, origin_y: float, origin_z: float):
        self.triangles = triangles
        self.x = x
        self.y = y
        self.z = z
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.rotate_x = rotate_x
        self.rotate_y = rotate_y
        self.rotate_z = rotate_z
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z

    @classmethod
    def from_json(cls, fobj: TextIO, **kwargs):
        raw = json.load(fobj)
        triangles = []
        for triangle in raw:
            triangles.append(Triangle(vertices=[Vertex(**vertex) for vertex in triangle]))
        return cls(
            triangles=triangles,
            **kwargs
        )

    def __iter__(self):
        return iter(self.triangles)


class Light(NamedTuple):
    r: float
    g: float
    b: float
    intensity: float


class DirectedLight(NamedTuple):
    r: float
    g: float
    b: float
    intensity: float
    x: float
    y: float
    z: float


class Camera(NamedTuple):
    x: float
    y: float
    z: float


class Edge(NamedTuple):
    # position
    x: int
    # color
    r: float
    g: float
    b: float
    a: float
    # depth buffer position
    z: float
    # normal vector
    nx: float
    ny: float
    nz: float


class Span:
    def __init__(self):
        self.edges = []

    @property
    def healthy(self):
        return len(self.edges) == 2

    @property
    def left_edge(self):
        return self.edges[0] if self.edges[0].x < self.edges[1].x else self.edges[1]

    @property
    def right_edge(self):
        return self.edges[0] if self.edges[0].x > self.edges[1].x else self.edges[1]


class Stepper:
    def __init__(self, diff, initial):
        self.diff = diff
        self.pos = initial

    @classmethod
    def from_start_end_length(cls, start, end, length):
        return cls((end - start) / length, start)

    def advance(self):
        self.pos += self.diff


class Spans:
    def __init__(self, size):
        self.spans = [Span() for _ in range(size)]
        self.first = None
        self.last = None

    def get(self, index: int):
        return self.spans[index]

    def add_edge(self, one: Vertex, two: Vertex):
        y_diff = math.ceil(one.y - 0.5) - math.ceil(two.y - 0.5)
        # degenerate edge
        if y_diff == 0:
            return
        start, end = (one, two) if y_diff > 0 else (two, one)
        length = abs(y_diff)
        y_pos = int(math.ceil(start.y - 0.5))
        y_end = int(math.ceil(end.y - 0.5))
        x_step = float(end.x - start.x) / length
        x_stepper = Stepper(x_step, start.x + (x_step / 2))
        z_step = (end.z - start.z) / length
        z_stepper = Stepper(z_step, start.z + (z_step / 2))

        r_stepper = Stepper.from_start_end_length(start.r, end.r, length)
        g_stepper = Stepper.from_start_end_length(start.g, end.g, length)
        b_stepper = Stepper.from_start_end_length(start.b, end.b, length)
        a_stepper = Stepper.from_start_end_length(start.a, end.a, length)
        nx_stepper = Stepper.from_start_end_length(start.nx, end.nx, length)
        ny_stepper = Stepper.from_start_end_length(start.ny, end.ny, length)
        nz_stepper = Stepper.from_start_end_length(start.nz, end.nz, length)
        for y in range(y_pos, y_end - 1, -1):
            x = int(math.ceil(x_stepper.pos))
            if y >= 0 and y < int(HEIGHT):
                if self.first is None or y < self.first:
                    self.first = y
                if self.last is None or y > self.last:
                    self.last = y
                self.spans[y].edges.append(Edge(
                    x=x,
                    r=r_stepper.pos,
                    g=g_stepper.pos,
                    b=b_stepper.pos,
                    a=a_stepper.pos,
                    z=z_stepper.pos,
                    nx=nx_stepper.pos,
                    ny=ny_stepper.pos,
                    nz=nz_stepper.pos,
                ))
            x_stepper.advance()
            r_stepper.advance()
            g_stepper.advance()
            b_stepper.advance()
            a_stepper.advance()
            z_stepper.advance()
            nx_stepper.advance()
            ny_stepper.advance()
            nz_stepper.advance()


def f2i(f: float) -> int:
    """
    Convert a float between 0.0 and 1.0 to an integer between 0 and 255.
    """
    f = max(0.0, min(1.0, f))
    if f == 1.0:
        return 255
    else:
        return int(math.floor(f * 256.0))


class Engine:
    def __init__(self, size):
        pygame.init()
        self.screen = pygame.display.set_mode(size)

    def rgba2c(self, r: float, g: float, b: float, a: float) -> pygame.Color:
        """
        Convert r,g,b,a (as floats) to a pygame Color
        """
        return pygame.Color(f2i(r), f2i(g), f2i(b), f2i(a))

    def init_frame(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def fill(self, r: float, g: float, b: float, a: float):
        self.screen.fill(self.rgba2c(r, g, b, a))

    def set_pixel(self, x: int, y: int, r: float, g: float, b: float, a: float):
        self.screen.set_at((x, y), self.rgba2c(r, g, b, a))

    def finish_frame(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()


def rotate(rotation: float, a: float, b: float) -> Tuple[float, float]:
    return (
        math.cos(rotation) * a + math.sin(rotation) * b,
        -math.sin(rotation) * a + math.cos(rotation) * b
    )


class World:
    def __init__(self, model: Model, ambient: Light, diffuse: DirectedLight, camera: Camera, engine: Engine):
        self.engine = engine
        self.depth_buffer = [float('Inf')] * WIDTH * HEIGHT
        self.model = model
        self.ambient = ambient
        self.diffuse = diffuse
        self.camera = camera

    def render(self):
        self.engine.fill(*BACKGROUND)
        self.depth_buffer = [float('Inf')] * WIDTH * HEIGHT

        for triangle in self.transform_and_project():
            self.draw(triangle)

        self.engine.finish_frame()

    def transform_and_project(self) -> Iterator[Triangle]:
        for triangle in self.model:
            yield Triangle(
                vertices=[self.transform_vertex(vertex) for vertex in triangle.vertices]
            )

    def transform_vertex(self, vertex: Vertex) -> Vertex:
        # Local origin
        x = vertex.x - self.model.origin_x
        y = vertex.y - self.model.origin_y
        z = vertex.z - self.model.origin_z
        # Scale
        x *= self.model.scale_x
        y *= self.model.scale_y
        z *= self.model.scale_z
        # Rotate
        y, z = rotate(self.model.rotate_x, y, z)
        x, z = rotate(self.model.rotate_y, x, z)
        x, y = rotate(self.model.rotate_z, x, y)
        # Normal Vector
        nx, nz = rotate(self.model.rotate_y, vertex.nx, vertex.nz)
        ny = vertex.ny
        # Translate
        x += self.model.origin_x
        y += self.model.origin_y
        z += self.model.origin_z
        # Camera Space
        x -= self.camera.x
        y -= self.camera.y
        z -= self.camera.z
        # 3D to 2D
        x /= (z + 100) * 0.01
        y /= (z + 100) * 0.01
        # Units to Pixels
        x *= float(HEIGHT) / UNIT_SCALE
        y *= float(HEIGHT) / UNIT_SCALE
        # Center screen
        x += float(WIDTH) / 2
        y += float(HEIGHT) / 2
        return Vertex(x=x, y=y, z=z, r=vertex.r, g=vertex.g, b=vertex.b, a=vertex.a, nx=nx, ny=ny, nz=nz)

    def draw(self, triangle: Triangle):
        # Check if the triangle is at least partially visible
        if not any(map(self.partially_inside_viewport, triangle.vertices)):
            return
        spans = Spans(HEIGHT)
        spans.add_edge(triangle.vertices[0], triangle.vertices[1])
        spans.add_edge(triangle.vertices[1], triangle.vertices[2])
        spans.add_edge(triangle.vertices[2], triangle.vertices[0])
        self.draw_spans(spans)

    def draw_spans(self, spans: Spans):
        if spans.last is None:
            return
        for y in range(spans.first, spans.last + 1):
            span = spans.get(y)
            if not span.healthy:
                continue

            left, right = span.left_edge, span.right_edge
            try:
                stepper = Stepper(1 / float(right.x - left.x), 0.0)
            except ZeroDivisionError:
                continue
            for x in range(left.x, right.x):
                # interpolate
                r = left.r + (right.r - left.r) * stepper.pos
                g = left.g + (right.g - left.g) * stepper.pos
                b = left.b + (right.b - left.b) * stepper.pos
                a = left.a + (right.a - left.a) * stepper.pos
                nx = left.nz + (right.nx - left.nx) * stepper.pos
                ny = left.ny + (right.ny - left.ny) * stepper.pos
                nz = left.nz + (right.nz - left.nz) * stepper.pos

                # check depth buffer
                should_draw = True
                z = left.z + (right.z - left.z) * stepper.pos
                offset = x + y * WIDTH
                if self.depth_buffer[offset] > z:
                    self.depth_buffer[offset] = z
                else:
                    should_draw = False

                if should_draw:
                    factor = min(max(0.0, -1 * (nx * self.diffuse.x + ny * self.diffuse.y + nz * self.diffuse.z)), 1.0)
                    r *= (self.ambient.r * self.ambient.intensity + factor * self.diffuse.r * self.diffuse.intensity)
                    g *= (self.ambient.g * self.ambient.intensity + factor * self.diffuse.g * self.diffuse.intensity)
                    b *= (self.ambient.b * self.ambient.intensity + factor * self.diffuse.b * self.diffuse.intensity)
                    r = max(min(r, 1.0), 0.0)
                    g = max(min(g, 1.0), 0.0)
                    b = max(min(b, 1.0), 0.0)
                    self.engine.set_pixel(x, y, r, g, b, a)
                stepper.advance()

    def partially_inside_viewport(self, vertex: Vertex) -> bool:
        return vertex.x >= 0 or vertex.x < float(WIDTH) or vertex.y >= 0 or vertex.y < float(HEIGHT)
