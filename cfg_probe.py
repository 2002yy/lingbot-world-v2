from wan.configs import WAN_CONFIGS
c = WAN_CONFIGS["i2v-1.3B"]
print("sample_fps =", getattr(c, "sample_fps", None))
print("num_frames =", getattr(c, "num_frames", None))
print("frame_num  =", getattr(c, "frame_num", None))
