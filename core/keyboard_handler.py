# core/keyboard_handler.py
import cv2 as cv

def handle_keyboard(key, ball_target, drone_target, plot_ball_file, plot_drone_file):
    if key == 27:  # ESC
        return "EXIT"
    elif key in (ord('r'), ord('R')):
        ball_target.is_recording = not ball_target.is_recording
        if not ball_target.is_recording:
            ball_target.trajectory.save_plot(plot_ball_file)
    elif key in (ord('t'), ord('T')):
        drone_target.is_recording = not drone_target.is_recording
        if not drone_target.is_recording:
            drone_target.trajectory.save_plot(plot_drone_file)
    elif key in (ord('s'), ord('S')):
        ball_target.trajectory.save_plot(plot_ball_file)
        drone_target.trajectory.save_plot(plot_drone_file)
    elif key in (ord('c'), ord('C')):
        ball_target.trajectory.clear()
    elif key in (ord('x'), ord('X')):
        drone_target.trajectory.clear()
    elif key in (ord('p'), ord('P')):
        return "TOGGLE_PAUSE"
    return "CONTINUE"