from __future__ import annotations

import colorsys
import dataclasses
import threading
import time
from pathlib import Path

import numpy as np
import scipy
import splines
import splines.quaternion
import viser
import viser.transforms as tf


@dataclasses.dataclass
class Keyframe(object):
    position: np.ndarray
    wxyz: np.ndarray
    override_fov_enabled: bool
    override_fov_rad: float
    aspect: float
    override_transition_enabled: bool
    override_transition_sec: float | None

    @staticmethod
    def from_camera(camera: viser.CameraHandle, aspect: float) -> Keyframe:
        return Keyframe(
            camera.position,
            camera.wxyz,
            override_fov_enabled=False,
            override_fov_rad=camera.fov,
            aspect=aspect,
            override_transition_enabled=False,
            override_transition_sec=None,
        )


class CameraPath(object):
    def __init__(
        self,
        server: viser.ViserServer,
        duration_element: viser.GuiInputHandle[float],
        scene_scale: float,
        scene_node_prefix: str = "/",
    ):
        self._server = server
        self._keyframes: dict[int, tuple[Keyframe, viser.CameraFrustumHandle]] = {}
        self._keyframe_counter: int = 0
        self._spline_nodes: list[viser.SceneNodeHandle] = []
        self._camera_edit_panel: viser.Gui3dContainerHandle | None = None

        self._orientation_spline: splines.quaternion.KochanekBartels | None = None
        self._position_spline: splines.KochanekBartels | None = None
        self._fov_spline: splines.KochanekBartels | None = None

        self._keyframes_visible: bool = True

        self._duration_element = duration_element
        self._scene_node_prefix = scene_node_prefix

        self.scene_scale = scene_scale
        # These parameters should be overridden externally.
        self.loop: bool = False
        self.framerate: float = 30.0
        self.tension: float = 0.5  # Tension / alpha term.
        self.default_fov: float = 0.0
        self.default_transition_sec: float = 0.0
        self.show_spline: bool = True

    def set_keyframes_visible(self, visible: bool) -> None:
        self._keyframes_visible = visible
        for keyframe in self._keyframes.values():
            keyframe[1].visible = visible

    def add_camera(self, keyframe: Keyframe, keyframe_index: int | None = None) -> None:
        """Add a new camera, or replace an old one if `keyframe_index` is passed in."""
        server = self._server

        # Add a keyframe if we aren't replacing an existing one.
        if keyframe_index is None:
            keyframe_index = self._keyframe_counter
            self._keyframe_counter += 1

        print(
            f"{keyframe.wxyz=} {keyframe.position=} {keyframe_index=} {keyframe.aspect=}"
        )
        frustum_handle = server.scene.add_camera_frustum(
            str(Path(self._scene_node_prefix) / f"cameras/{keyframe_index}"),
            fov=(
                keyframe.override_fov_rad
                if keyframe.override_fov_enabled
                else self.default_fov
            ),
            aspect=keyframe.aspect,
            scale=0.1 * self.scene_scale,
            color=(200, 10, 30),
            wxyz=keyframe.wxyz,
            position=keyframe.position,
            visible=self._keyframes_visible,
        )
        self._server.scene.add_icosphere(
            str(Path(self._scene_node_prefix) / f"cameras/{keyframe_index}/sphere"),
            radius=0.03,
            color=(200, 10, 30),
        )

        @frustum_handle.on_click
        def _(_) -> None:
            if self._camera_edit_panel is not None:
                self._camera_edit_panel.remove()
                self._camera_edit_panel = None

            with server.scene.add_3d_gui_container(
                "/camera_edit_panel",
                position=keyframe.position,
            ) as camera_edit_panel:
                self._camera_edit_panel = camera_edit_panel
                override_fov = server.gui.add_checkbox(
                    "Override FOV", initial_value=keyframe.override_fov_enabled
                )
                override_fov_degrees = server.gui.add_slider(
                    "Override FOV (degrees)",
                    5.0,
                    175.0,
                    step=0.1,
                    initial_value=keyframe.override_fov_rad * 180.0 / np.pi,
                    disabled=not keyframe.override_fov_enabled,
                )
                delete_button = server.gui.add_button(
                    "Delete", color="red", icon=viser.Icon.TRASH
                )
                go_to_button = server.gui.add_button("Go to")
                close_button = server.gui.add_button("Close")

            @override_fov.on_update
            def _(_) -> None:
                keyframe.override_fov_enabled = override_fov.value
                override_fov_degrees.disabled = not override_fov.value
                self.add_camera(keyframe, keyframe_index)

            @override_fov_degrees.on_update
            def _(_) -> None:
                keyframe.override_fov_rad = override_fov_degrees.value / 180.0 * np.pi
                self.add_camera(keyframe, keyframe_index)

            @delete_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                with event.client.gui.add_modal("Confirm") as modal:
                    event.client.gui.add_markdown("Delete keyframe?")
                    confirm_button = event.client.gui.add_button(
                        "Yes", color="red", icon=viser.Icon.TRASH
                    )
                    exit_button = event.client.gui.add_button("Cancel")

                    @confirm_button.on_click
                    def _(_) -> None:
                        assert camera_edit_panel is not None

                        keyframe_id = None
                        for i, keyframe_tuple in self._keyframes.items():
                            if keyframe_tuple[1] is frustum_handle:
                                keyframe_id = i
                                break
                        assert keyframe_id is not None

                        self._keyframes.pop(keyframe_id)
                        frustum_handle.remove()
                        camera_edit_panel.remove()
                        self._camera_edit_panel = None
                        modal.close()
                        self.update_spline()

                    @exit_button.on_click
                    def _(_) -> None:
                        modal.close()

            @go_to_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                client = event.client
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(keyframe.wxyz), keyframe.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(10):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 9.0
                    )

                    # Important bit: we atomically set both the orientation and the position
                    # of the camera.
                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                    time.sleep(1.0 / 30.0)

            @close_button.on_click
            def _(_) -> None:
                assert camera_edit_panel is not None
                camera_edit_panel.remove()
                self._camera_edit_panel = None

        self._keyframes[keyframe_index] = (keyframe, frustum_handle)

    def update_aspect(self, aspect: float) -> None:
        for keyframe_index, frame in self._keyframes.items():
            frame = dataclasses.replace(frame[0], aspect=aspect)
            self.add_camera(frame, keyframe_index=keyframe_index)

    def get_aspect(self) -> float:
        """Get W/H aspect ratio, which is shared across all keyframes."""
        assert len(self._keyframes) > 0
        return next(iter(self._keyframes.values()))[0].aspect

    def reset(self) -> None:
        for frame in self._keyframes.values():
            print(f"removing {frame[1]}")
            frame[1].remove()
        self._keyframes.clear()
        self.update_spline()
        print("camera path reset")

    def spline_t_from_t_sec(self, time: np.ndarray) -> np.ndarray:
        """From a time value in seconds, compute a t value for our geometric
        spline interpolation. An increment of 1 for the latter will move the
        camera forward by one keyframe.

        We use a PCHIP spline here to guarantee monotonicity.
        """
        transition_times_cumsum = self.compute_transition_times_cumsum()
        spline_indices = np.arange(transition_times_cumsum.shape[0])

        if self.loop:
            # In the case of a loop, we pad the spline to match the start/end
            # slopes.
            interpolator = scipy.interpolate.PchipInterpolator(
                x=np.concatenate(
                    [
                        [-(transition_times_cumsum[-1] - transition_times_cumsum[-2])],
                        transition_times_cumsum,
                        transition_times_cumsum[-1:] + transition_times_cumsum[1:2],
                    ],
                    axis=0,
                ),
                y=np.concatenate(
                    [[-1], spline_indices, [spline_indices[-1] + 1]],  # type: ignore
                    axis=0,
                ),
            )
        else:
            interpolator = scipy.interpolate.PchipInterpolator(
                x=transition_times_cumsum, y=spline_indices
            )

        # Clip to account for floating point error.
        return np.clip(interpolator(time), 0, spline_indices[-1])

    def interpolate_pose_and_fov_rad(
        self, normalized_t: float
    ) -> tuple[tf.SE3, float] | None:
        if len(self._keyframes) < 2:
            return None

        self._fov_spline = splines.KochanekBartels(
            [
                (
                    keyframe[0].override_fov_rad
                    if keyframe[0].override_fov_enabled
                    else self.default_fov
                )
                for keyframe in self._keyframes.values()
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        assert self._orientation_spline is not None
        assert self._position_spline is not None
        assert self._fov_spline is not None

        max_t = self.compute_duration()
        t = max_t * normalized_t
        spline_t = float(self.spline_t_from_t_sec(np.array(t)))

        quat = self._orientation_spline.evaluate(spline_t)
        assert isinstance(quat, splines.quaternion.UnitQuaternion)
        return (
            tf.SE3.from_rotation_and_translation(
                tf.SO3(np.array([quat.scalar, *quat.vector])),
                self._position_spline.evaluate(spline_t),
            ),
            float(self._fov_spline.evaluate(spline_t)),
        )

    def update_spline(self) -> None:
        num_frames = int(self.compute_duration() * self.framerate)
        keyframes = list(self._keyframes.values())

        if num_frames <= 0 or not self.show_spline or len(keyframes) < 2:
            for node in self._spline_nodes:
                node.remove()
            self._spline_nodes.clear()
            return

        transition_times_cumsum = self.compute_transition_times_cumsum()

        self._orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(
                    np.roll(keyframe[0].wxyz, shift=-1)
                )
                for keyframe in keyframes
            ],
            tcb=(self.tension, 0.0, 0.0),
            # alpha=self.tension,
            endconditions="closed" if self.loop else "natural",
        )
        self._position_spline = splines.KochanekBartels(
            [keyframe[0].position for keyframe in keyframes],
            tcb=(self.tension, 0.0, 0.0),
            # alpha=self.tension,
            endconditions="closed" if self.loop else "natural",
        )

        # Update visualized spline.
        points_array = self._position_spline.evaluate(
            self.spline_t_from_t_sec(
                np.linspace(0, transition_times_cumsum[-1], num_frames)
            )
        )
        colors_array = np.array(
            [
                colorsys.hls_to_rgb(h, 0.5, 1.0)
                for h in np.linspace(0.0, 1.0, len(points_array))
            ]
        )

        # Clear prior spline nodes.
        for node in self._spline_nodes:
            node.remove()
        self._spline_nodes.clear()

        self._spline_nodes.append(
            self._server.scene.add_spline_catmull_rom(
                str(Path(self._scene_node_prefix) / "camera_spline"),
                positions=points_array,
                color=(220, 220, 220),
                closed=self.loop,
                line_width=1.0,
                segments=points_array.shape[0] + 1,
            )
        )
        self._spline_nodes.append(
            self._server.scene.add_point_cloud(
                str(Path(self._scene_node_prefix) / "camera_spline/points"),
                points=points_array,
                colors=colors_array,
                point_size=0.04,
            )
        )

        def make_transition_handle(i: int) -> None:
            assert self._position_spline is not None
            transition_pos = self._position_spline.evaluate(
                float(
                    self.spline_t_from_t_sec(
                        (transition_times_cumsum[i] + transition_times_cumsum[i + 1])
                        / 2.0,
                    )
                )
            )
            transition_sphere = self._server.scene.add_icosphere(
                str(Path(self._scene_node_prefix) / f"camera_spline/transition_{i}"),
                radius=0.04,
                color=(255, 0, 0),
                position=transition_pos,
            )
            self._spline_nodes.append(transition_sphere)

            @transition_sphere.on_click
            def _(_) -> None:
                server = self._server

                if self._camera_edit_panel is not None:
                    self._camera_edit_panel.remove()
                    self._camera_edit_panel = None

                keyframe_index = (i + 1) % len(self._keyframes)
                keyframe = keyframes[keyframe_index][0]

                with server.scene.add_3d_gui_container(
                    "/camera_edit_panel",
                    position=transition_pos,
                ) as camera_edit_panel:
                    self._camera_edit_panel = camera_edit_panel
                    override_transition_enabled = server.gui.add_checkbox(
                        "Override transition",
                        initial_value=keyframe.override_transition_enabled,
                    )
                    override_transition_sec = server.gui.add_number(
                        "Override transition (sec)",
                        initial_value=(
                            keyframe.override_transition_sec
                            if keyframe.override_transition_sec is not None
                            else self.default_transition_sec
                        ),
                        min=0.001,
                        max=30.0,
                        step=0.001,
                        disabled=not override_transition_enabled.value,
                    )
                    close_button = server.gui.add_button("Close")

                @override_transition_enabled.on_update
                def _(_) -> None:
                    keyframe.override_transition_enabled = (
                        override_transition_enabled.value
                    )
                    override_transition_sec.disabled = (
                        not override_transition_enabled.value
                    )
                    self._duration_element.value = self.compute_duration()

                @override_transition_sec.on_update
                def _(_) -> None:
                    keyframe.override_transition_sec = override_transition_sec.value
                    self._duration_element.value = self.compute_duration()

                @close_button.on_click
                def _(_) -> None:
                    assert camera_edit_panel is not None
                    camera_edit_panel.remove()
                    self._camera_edit_panel = None

        (num_transitions_plus_1,) = transition_times_cumsum.shape
        for i in range(num_transitions_plus_1 - 1):
            make_transition_handle(i)

    def compute_duration(self) -> float:
        """Compute the total duration of the trajectory."""
        total = 0.0
        for i, (keyframe, frustum) in enumerate(self._keyframes.values()):
            if i == 0 and not self.loop:
                continue
            del frustum
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled
                and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
        return total

    def compute_transition_times_cumsum(self) -> np.ndarray:
        """Compute the total duration of the trajectory."""
        total = 0.0
        out = [0.0]
        for i, (keyframe, frustum) in enumerate(self._keyframes.values()):
            if i == 0:
                continue
            del frustum
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled
                and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
            out.append(total)

        if self.loop:
            keyframe = next(iter(self._keyframes.values()))[0]
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled
                and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
            out.append(total)

        return np.array(out)


@dataclasses.dataclass
class GuiState:
    preview_render: bool
    preview_fov: float
    preview_aspect: float
    camera_path_list: list | None
    active_input_index: int


def define_gui(
    server: viser.ViserServer,
    init_fov: float = 75.0,
    img_wh: tuple[int, int] = (576, 576),
    **kwargs,
) -> GuiState:
    gui_state = GuiState(
        preview_render=False,
        preview_fov=0.0,
        preview_aspect=1.0,
        camera_path_list=None,
        active_input_index=0,
    )

    with server.gui.add_folder("Advanced", expand_by_default=False, order=100):
        transition_sec_number = server.gui.add_number(
            "Transition (sec)",
            min=0.001,
            max=30.0,
            step=0.001,
            initial_value=1.5,
            hint="Time in seconds between each keyframe, which can also be overridden on a per-transition basis.",
        )
        framerate_number = server.gui.add_number(
            "FPS", min=0.1, max=240.0, step=1e-2, initial_value=30.0
        )
        framerate_buttons = server.gui.add_button_group("", ("24", "30", "60"))
        duration_number = server.gui.add_number(
            "Duration (sec)",
            min=0.0,
            max=1e8,
            step=0.001,
            initial_value=0.0,
            disabled=True,
        )

        @framerate_buttons.on_click
        def _(_) -> None:
            framerate_number.value = float(framerate_buttons.value)

    fov_degree_slider = server.gui.add_slider(
        "FOV",
        initial_value=init_fov,
        min=0.1,
        max=175.0,
        step=0.01,
        hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
    )

    @fov_degree_slider.on_update
    def _(_) -> None:
        fov_radians = fov_degree_slider.value / 180.0 * np.pi
        for client in server.get_clients().values():
            client.camera.fov = fov_radians
        camera_path.default_fov = fov_radians

        # Updating the aspect ratio will also re-render the camera frustums.
        # Could rethink this.
        camera_path.update_aspect(img_wh[0] / img_wh[1])
        compute_and_update_preview_camera_state()

    scene_node_prefix = "/render_assets"
    base_scene_node = server.scene.add_frame(scene_node_prefix, show_axes=False)
    add_button = server.gui.add_button(
        "Add keyframe",
        icon=viser.Icon.PLUS,
        hint="Add a new keyframe at the current pose.",
    )

    @add_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client_id is not None
        camera = server.get_clients()[event.client_id].camera
        pose = tf.SE3.from_rotation_and_translation(
            tf.SO3(camera.wxyz), camera.position
        )
        print(f"client {event.client_id} at {camera.position} {camera.wxyz}")
        print(f"camera pose {pose.as_matrix()}")

        # Add this camera to the path.
        camera_path.add_camera(
            Keyframe.from_camera(
                camera,
                aspect=img_wh[0] / img_wh[1],
            ),
        )
        duration_number.value = camera_path.compute_duration()
        camera_path.update_spline()

    clear_keyframes_button = server.gui.add_button(
        "Clear keyframes",
        icon=viser.Icon.TRASH,
        hint="Remove all keyframes from the render path.",
    )

    @clear_keyframes_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client_id is not None
        client = server.get_clients()[event.client_id]
        with client.atomic(), client.gui.add_modal("Confirm") as modal:
            client.gui.add_markdown("Clear all keyframes?")
            confirm_button = client.gui.add_button(
                "Yes", color="red", icon=viser.Icon.TRASH
            )
            exit_button = client.gui.add_button("Cancel")

            @confirm_button.on_click
            def _(_) -> None:
                camera_path.reset()
                modal.close()

                duration_number.value = camera_path.compute_duration()

                nonlocal gui_state
                gui_state.camera_path_list = None

                # Clear move handles.
                if len(transform_controls) > 0:
                    for t in transform_controls:
                        t.remove()
                    transform_controls.clear()
                    return

            @exit_button.on_click
            def _(_) -> None:
                modal.close()

    play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
    pause_button = server.gui.add_button(
        "Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False
    )

    # Play the camera trajectory when the play button is pressed.
    @play_button.on_click
    def _(_) -> None:
        play_button.visible = False
        pause_button.visible = True

        def play() -> None:
            while not play_button.visible:
                max_frame = int(framerate_number.value * duration_number.value)
                if max_frame > 0:
                    assert preview_frame_slider is not None
                    preview_frame_slider.value = (
                        preview_frame_slider.value + 1
                    ) % max_frame
                time.sleep(1.0 / framerate_number.value)

        threading.Thread(target=play).start()

    # Play the camera trajectory when the play button is pressed.
    @pause_button.on_click
    def _(_) -> None:
        play_button.visible = True
        pause_button.visible = False

    preview_render_button = server.gui.add_button(
        "Preview render", hint="Show a preview of the render in the viewport."
    )
    preview_render_stop_button = server.gui.add_button(
        "Exit Render Preview", color="red", visible=False
    )

    @preview_render_button.on_click
    def _(_) -> None:
        gui_state.preview_render = True
        preview_render_button.visible = False
        preview_render_stop_button.visible = True

        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return
        pose, fov = maybe_pose_and_fov_rad
        del fov

        # Hide all render assets when we're previewing the render.
        nonlocal base_scene_node
        base_scene_node.visible = False

        # Back up and then set camera poses.
        for client in server.get_clients().values():
            camera_pose_backup_from_id[client.client_id] = (
                client.camera.position,
                client.camera.look_at,
                client.camera.up_direction,
            )
            client.camera.wxyz = pose.rotation().wxyz
            client.camera.position = pose.translation()

    @preview_render_stop_button.on_click
    def _(_) -> None:
        gui_state.preview_render = False
        preview_render_button.visible = True
        preview_render_stop_button.visible = False

        # Revert camera poses.
        for client in server.get_clients().values():
            if client.client_id not in camera_pose_backup_from_id:
                continue
            cam_position, cam_look_at, cam_up = camera_pose_backup_from_id.pop(
                client.client_id
            )
            client.camera.position = cam_position
            client.camera.look_at = cam_look_at
            client.camera.up_direction = cam_up
            client.flush()

        # Un-hide render assets.
        nonlocal base_scene_node
        base_scene_node.visible = True

    def get_max_frame_index() -> int:
        return max(1, int(framerate_number.value * duration_number.value) - 1)

    def add_preview_frame_slider() -> viser.GuiInputHandle[int] | None:
        """Helper for creating the current frame # slider. This is removed and
        re-added anytime the `max` value changes."""

        preview_frame_slider = server.gui.add_slider(
            "Preview frame",
            min=0,
            max=get_max_frame_index(),
            step=1,
            initial_value=0,
            order=save_path_button.order + 0.01,
            disabled=get_max_frame_index() == 1,
        )
        play_button.disabled = preview_frame_slider.disabled
        preview_render_button.disabled = preview_frame_slider.disabled
        save_path_button.disabled = preview_frame_slider.disabled

        @preview_frame_slider.on_update
        def _(_) -> None:
            nonlocal preview_camera_handle
            maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
            if maybe_pose_and_fov_rad is None:
                return
            pose, fov_rad = maybe_pose_and_fov_rad

            preview_camera_handle = server.scene.add_camera_frustum(
                str(Path(scene_node_prefix) / "preview_camera"),
                fov=fov_rad,
                aspect=img_wh[0] / img_wh[1],
                scale=0.35,
                wxyz=pose.rotation().wxyz,
                position=pose.translation(),
                color=(10, 200, 30),
            )
            if gui_state.preview_render:
                for client in server.get_clients().values():
                    client.camera.wxyz = pose.rotation().wxyz
                    client.camera.position = pose.translation()

        return preview_frame_slider

    save_path_button = server.gui.add_button(
        "Set camera path",
        color="green",
        icon=viser.Icon.CHECK,
        hint="Save the camera path to json.",
    )

    @save_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        num_frames = int(framerate_number.value * duration_number.value)

        def get_intrinsics(W, H, fov_rad):
            focal = 0.5 * H / np.tan(0.5 * fov_rad)
            return np.array(
                [[focal, 0.0, 0.5 * W], [0.0, focal, 0.5 * H], [0.0, 0.0, 1.0]]
            )

        camera_path_list = []
        for i in range(num_frames):
            maybe_pose_and_fov_rad = camera_path.interpolate_pose_and_fov_rad(
                i / num_frames
            )
            if maybe_pose_and_fov_rad is None:
                return
            pose, fov_rad = maybe_pose_and_fov_rad
            H = img_wh[1]
            W = img_wh[0]
            K = get_intrinsics(W, H, fov_rad)
            w2c = pose.inverse().as_matrix()
            camera_path_list.append(
                {
                    "w2c": w2c.flatten().tolist(),
                    "K": K.flatten().tolist(),
                    "img_wh": (W, H),
                }
            )
        nonlocal gui_state
        gui_state.camera_path_list = camera_path_list

    preview_frame_slider = add_preview_frame_slider()

    loop = server.gui.add_checkbox(
        "Loop", False, hint="Add a segment between the first and last keyframes."
    )

    @loop.on_update
    def _(_) -> None:
        camera_path.loop = loop.value
        duration_number.value = camera_path.compute_duration()

    tension_slider = server.gui.add_slider(
        "Spline tension",
        min=0.0,
        max=1.0,
        initial_value=0.0,
        step=0.01,
        hint="Tension parameter for adjusting smoothness of spline interpolation.",
    )

    @tension_slider.on_update
    def _(_) -> None:
        camera_path.tension = tension_slider.value
        camera_path.update_spline()

    move_checkbox = server.gui.add_checkbox(
        "Move keyframes",
        initial_value=False,
        hint="Toggle move handles for keyframes in the scene.",
    )

    transform_controls: list[viser.SceneNodeHandle] = []

    @move_checkbox.on_update
    def _(event: viser.GuiEvent) -> None:
        # Clear move handles when toggled off.
        if move_checkbox.value is False:
            for t in transform_controls:
                t.remove()
            transform_controls.clear()
            return

        def _make_transform_controls_callback(
            keyframe: tuple[Keyframe, viser.SceneNodeHandle],
            controls: viser.TransformControlsHandle,
        ) -> None:
            @controls.on_update
            def _(_) -> None:
                keyframe[0].wxyz = controls.wxyz
                keyframe[0].position = controls.position

                keyframe[1].wxyz = controls.wxyz
                keyframe[1].position = controls.position

                camera_path.update_spline()

        # Show move handles.
        assert event.client is not None
        for keyframe_index, keyframe in camera_path._keyframes.items():
            controls = event.client.scene.add_transform_controls(
                str(Path(scene_node_prefix) / f"keyframe_move/{keyframe_index}"),
                scale=0.4,
                wxyz=keyframe[0].wxyz,
                position=keyframe[0].position,
            )
            transform_controls.append(controls)
            _make_transform_controls_callback(keyframe, controls)

    show_keyframe_checkbox = server.gui.add_checkbox(
        "Show keyframes",
        initial_value=True,
        hint="Show keyframes in the scene.",
    )

    @show_keyframe_checkbox.on_update
    def _(_: viser.GuiEvent) -> None:
        camera_path.set_keyframes_visible(show_keyframe_checkbox.value)

    show_spline_checkbox = server.gui.add_checkbox(
        "Show spline",
        initial_value=True,
        hint="Show camera path spline in the scene.",
    )

    @show_spline_checkbox.on_update
    def _(_) -> None:
        camera_path.show_spline = show_spline_checkbox.value
        camera_path.update_spline()

    @transition_sec_number.on_update
    def _(_) -> None:
        camera_path.default_transition_sec = transition_sec_number.value
        duration_number.value = camera_path.compute_duration()

    preview_camera_handle: viser.SceneNodeHandle | None = None

    def remove_preview_camera() -> None:
        nonlocal preview_camera_handle
        if preview_camera_handle is not None:
            preview_camera_handle.remove()
            preview_camera_handle = None

    def compute_and_update_preview_camera_state() -> tuple[tf.SE3, float] | None:
        """Update the render tab state with the current preview camera pose.
        Returns current camera pose + FOV if available."""

        if preview_frame_slider is None:
            return None
        maybe_pose_and_fov_rad = camera_path.interpolate_pose_and_fov_rad(
            preview_frame_slider.value / get_max_frame_index()
        )
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return None
        pose, fov_rad = maybe_pose_and_fov_rad
        gui_state.preview_fov = fov_rad
        gui_state.preview_aspect = camera_path.get_aspect()
        return pose, fov_rad

    # We back up the camera poses before and after we start previewing renders.
    camera_pose_backup_from_id: dict[int, tuple] = {}

    # Update the # of frames.
    @duration_number.on_update
    @framerate_number.on_update
    def _(_) -> None:
        remove_preview_camera()  # Will be re-added when slider is updated.

        nonlocal preview_frame_slider
        old = preview_frame_slider
        assert old is not None

        preview_frame_slider = add_preview_frame_slider()
        if preview_frame_slider is not None:
            old.remove()
        else:
            preview_frame_slider = old

        camera_path.framerate = framerate_number.value
        camera_path.update_spline()

    camera_path = CameraPath(
        server,
        duration_number,
        scene_node_prefix=scene_node_prefix,
        **kwargs,
    )
    camera_path.default_fov = fov_degree_slider.value / 180.0 * np.pi
    camera_path.default_transition_sec = transition_sec_number.value

    return gui_state
