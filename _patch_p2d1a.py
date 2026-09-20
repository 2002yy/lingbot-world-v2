import py_compile

p = "/home/zhang/ai/lingbot-world-v2/p2d1a_exactness.py"
s = open(p).read()

s = s.replace("MAX_SAMPLES = 240", "MAX_SAMPLES = 48   # keep VRAM flat; 48 already\n"
                                   "                   # covers several blocks x all 4 forward kinds")

# store captures on CPU so the corpus does not consume VRAM during the run
s = s.replace("""                norm_out[name] = out.detach()""",
              """                norm_out[name] = out.detach().to("cpu")""")
s = s.replace("""                    CORPUS.append((norm_out.pop(key),
                                   ec[sc].squeeze(2).detach(),
                                   ec[si].squeeze(2).detach(),
                                   nm))""",
              """                    CORPUS.append((norm_out.pop(key),
                                   ec[sc].squeeze(2).detach().to("cpu"),
                                   ec[si].squeeze(2).detach().to("cpu"),
                                   nm))""")

# move each sample back to the GPU only while testing it
s = s.replace("""    for idx, (n, sc, sh, tag) in enumerate(CORPUS):
        with torch.no_grad():
            r = ref_chain(n, sc, sh)""",
              """    for idx, (n_c, sc_c, sh_c, tag) in enumerate(CORPUS):
        n = n_c.to(dev); sc = sc_c.to(dev); sh = sh_c.to(dev)
        with torch.no_grad():
            r = ref_chain(n, sc, sh)""")

s = s.replace("""    with torch.no_grad():
        r0 = ref_chain(*CORPUS[0][:3])
        r1 = ref_chain(*CORPUS[0][:3])""",
              """    with torch.no_grad():
        a0 = CORPUS[0][0].to(dev); a1 = CORPUS[0][1].to(dev); a2 = CORPUS[0][2].to(dev)
        r0 = ref_chain(a0, a1, a2)
        r1 = ref_chain(a0, a1, a2)""")

open(p, "w").write(s)
py_compile.compile(p, doraise=True)
print("patched + compiles OK")
