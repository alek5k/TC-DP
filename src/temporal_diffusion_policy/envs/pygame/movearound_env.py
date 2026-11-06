from typing import Tuple, Sequence
import numpy as np


import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
import math
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import collections
import skimage.transform as st
from scipy.ndimage import distance_transform_edt
from pymunk.pygame_util import to_pygame




#@markdown ### **Environment**
#@markdown
#@markdown **Goal**: push the gray T-block into the green area.
#@markdown
#@markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)



positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes, circle_resolution=32):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]  # Close the polygon
            geoms.append(sg.Polygon(verts))
        elif isinstance(shape, pymunk.shapes.Circle):
            center = body.local_to_world(shape.offset)
            angle_step = 2 * np.pi / circle_resolution
            verts = [(center.x + shape.radius * np.cos(i * angle_step),
                      center.y + shape.radius * np.sin(i * angle_step))
                     for i in range(circle_resolution)]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class MoveAround(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            render_size=144
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 200, 50    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # agent_pos
        self.observation_space = spaces.Dict({
            'full_image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pose': spaces.Box(
                low=np.array([0, 0,], dtype=np.float32),
                high=np.array([ws, ws], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Box(
            low=np.array([-ws, -ws], dtype=np.float32),
            high=np.array([ws, ws], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None

        self.step_count = 0  
        
        self.revolutions = 0
        self.total_angle_change = 0.0
        self.prev_angle = None
        self.around_obstacle_reward_given = False


    def reset(self):
        # Reset the environment to a new state.
        self.revolutions = 0
        self.total_angle_change = 0.0
        self.prev_angle = None
        self.around_obstacle_reward_given = False
        
        self.step_count = 0

        seed = self._seed
        self._setup()
        rs = np.random.RandomState(seed=seed)
        
        # Set random agent position and rotation
        agent_pos = rs.randint(40, 80, size=2)

        # Main goal position
        goal_pos = np.array([450, 450])

        state = np.array([
            agent_pos[0], agent_pos[1],  # agent position
            goal_pos[0], goal_pos[1],    # goal position
        ])
        self._set_state(state)

        obs = self._get_obs()
        return obs

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        self.step_count += 1

        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                # PD control
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        self.space.step(dt)  # Final physics step

        # --- Collision detection ---
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        
        # --- Geometry setup ---
        goal_body = self._get_goal_pose_body(self.goal_pos)
        goal_geom = pymunk_to_shapely(goal_body, self.goal.shapes)
        agent_geom = pymunk_to_shapely(self.agent, self.agent.shapes)
        
        # --- Intersection (goal reached) ---
        intersection = goal_geom.intersection(agent_geom).area
        agent_area = agent_geom.area
        coverage = intersection / agent_area

        # --- Angle tracking around obstacle ---
        obstacle_center = np.array(self.obstacle.position)
        agent_pos = np.array([self.agent.position.x, self.agent.position.y])
        vec = agent_pos - obstacle_center
        angle = np.arctan2(vec[1], vec[0])

        if self.prev_angle is not None:
            delta_angle = angle - self.prev_angle
            delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize
            self.total_angle_change += delta_angle

            if abs(self.total_angle_change) >= 2 * np.pi:
                self.revolutions += 1
                self.total_angle_change = 0.0
                print(f"🔁 Completed revolution #{self.revolutions}")


        self.prev_angle = angle

        # --- Reward & Completion Logic ---
        reward = 0.0
        done = False

        # Give reward only after 3 revolutions and goal is reached
        if self.revolutions >= 3 and coverage >= self.success_threshold:
            reward = 1.0  # Final reward
            done = True

        elif coverage >= self.success_threshold and self.revolutions < 3:
            print(f"⚠️ Goal reached too early, revolutions = {self.revolutions}")
            reward = 0.0  # No reward
            done = False  # Don't end the episode yet

        # --- Collision penalty (optional) ---
        if n_contact_points_per_step > 0:
            done = True
            reward = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    
    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act():
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        img = self._render_frame(mode='human')
        agent_pose = np.array([self.agent.position.x, self.agent.position.y], dtype=np.float32)
        obs = { 
            'full_image': np.moveaxis(img.astype(np.float32) / 255, -1, 0),
            'agent_pose': agent_pose,
            'goal_pose': np.array(self.goal_pos),
            'step_count': self.step_count,
        }
        ### Check shapes of observations
        # for k, v in obs.items():
        #     if hasattr(v, "shape"):
        #         print(f"{k}: {v.shape}")
        #     else:
        #         print(f"{k}: {type(v)}")
        return obs

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'n_contacts': n_contact_points_per_step,
            }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw first goal
        goal_body = self._get_goal_pose_body(self.goal_pos)
        for shape in self.goal.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, shape.color, goal_points)

        
        # Draw agent.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()

        self.agent.position = state[:2]
        self.goal_pos= state[2:4]

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((40, 40), 25)
        self.goal = self.add_goal((-150, -150), 100, 100) #fake goal shape
        self.goal_pos = [450, 450]

        # Add collision handeling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.obstacle = self.add_obstacle((256, 256), 30)

        self.success_threshold = 0.98   # 98% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose
        return body
    
    def add_circle(self, position, radius):
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, radius)

        body = pymunk.Body(mass, inertia, body_type=pymunk.Body.DYNAMIC)
        body.position = position

        shape = pymunk.Circle(body, radius)
        shape.friction = 1.0
        shape.color = pygame.Color('RoyalBlue')

        self.space.add(body, shape)

        return body
    
    def add_goal(self, position, width=40, height=40):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (width, height))
        # body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body = pymunk.Body(mass, inertia)
        body.position = position

        shape = pymunk.Poly.create_box(body, (width, height))
        shape.color = pygame.Color('LightGreen')
        self.space.add(body, shape)
        return body

    def add_obstacle(self, position, radius):
        mass = 1000
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('Red')
        self.space.add(body, shape)
        return body