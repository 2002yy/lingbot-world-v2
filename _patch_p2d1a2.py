import py_compile

p = "/home/zhang/ai/lingbot-world-v2/p2d1a2_matrix.py"
s = open(p).read()

old = """    def make_patch2(bi):
        def patched(self, x, e, *a, **kw):
            out = orig_cb(self, x, e, *a, **kw)
            if CAP["on"] and len(SAMPLES["mod1"]) < CAP["per_chain"]:"""
new = """    def make_patch2(bi, blk):
        # NOTE: this is assigned as `blk.forward = patched`, i.e. an INSTANCE
        # attribute, so it is NOT bound -- a leading `self` parameter would
        # swallow the real `x` argument. Capture the block by closure instead.
        def patched(x, e, *a, **kw):
            out = orig_cb(blk, x, e, *a, **kw)
            if CAP["on"] and len(SAMPLES["mod1"]) < CAP["per_chain"]:"""
assert old in s, "anchor1"
s = s.replace(old, new)

# fix the body references from `self.` to `blk.`
seg_start = s.index("        def patched(x, e, *a, **kw):")
seg_end = s.index("    for i, blk in enumerate(pipe.model.blocks):\n        blk.forward = make_patch2(i)")
body = s[seg_start:seg_end]
body = body.replace("self.modulation", "blk.modulation")
s = s[:seg_start] + body + s[seg_end:]

s = s.replace("        blk.forward = make_patch2(i)",
              "        blk.forward = make_patch2(i, blk)")

# remove the dead first `make_patch` definition entirely
a = s.find("    def make_patch(bi):")
if a != -1:
    b = s.find("    class _CamSpy:")
    if b == -1:
        b = s.find("    # capture cam_scale/cam_shift")
    s = s[:a] + s[b:]

open(p, "w").write(s)
py_compile.compile(p, doraise=True)
print("patched + compiles OK")
