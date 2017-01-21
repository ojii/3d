import pygame

from engine import World, Model, Light, DirectedLight, Camera, Engine, SIZE


def main():
    with open('model.json', 'r') as fobj:
        model = Model.from_json(
            fobj=fobj,
            x=0.0, y=0.0, z=0.0,
            scale_x=1.0, scale_y=1.0, scale_z=1.0,
            rotate_x=0.0, rotate_y=0.0, rotate_z=0.0,
            origin_x=10.0, origin_y=0.0, origin_z=10.0
        )

    camera = Camera(x=0.0, y=20.0, z=-20.0)

    ambient = Light(r=1.0, g=1.0, b=1.0, intensity=0.2)

    diffuse = DirectedLight(r=1.0, g=1.0, b=1.0, intensity=0.8, x=0.0, y=0.0, z=0.0)

    engine = Engine(SIZE)

    world = World(model=model, ambient=ambient, diffuse=diffuse, camera=camera, engine=engine)

    bounce_speed = 60.0
    bounce_accel = -60.0
    delta_time = 0.1

    try:
        clock = pygame.time.Clock()
        while True:
            if not engine.init_frame():
                break
            world.render()
            clock.tick_busy_loop(1)
            bounce_speed += bounce_accel * delta_time
            model.y += bounce_speed * delta_time
            if model.y < 0:
                model.y -= bounce_speed * delta_time
                bounce_speed = -bounce_speed
            model.scale_y = (model.y + 30.0) / 50.0
            model.scale_x = 1.5 * model.scale_y / 2.0
            model.scale_z = model.scale_x
            model.rotate_y += 1.5 * delta_time

    except KeyboardInterrupt:
        pass
    finally:
        engine.quit()

if __name__ == '__main__':
    main()
