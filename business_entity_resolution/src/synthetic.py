"""Tiny synthetic multilingual dataset (US/India train, +France test) for smoke-testing only."""
import os
import random
import sys

W = {
    "US": ["Blue", "Ridge", "Summit", "Liberty", "Oak", "Harbor", "Pioneer", "Eagle", "Lakeside", "Maple",
           "Cedar", "Union", "Atlas", "Granite", "Prairie"],
    "India": ["Shree", "Sai", "Bharat", "Lakshmi", "Ganesh", "Kumar", "Sharma", "Deccan", "Maruti",
              "Krishna", "Patel", "Gupta", "Om", "Vishwa", "Annapurna"],
    "France": ["Maison", "Lumiere", "Dupont", "Petit", "Chateau", "Belle", "Atelier", "Soleil", "Martin",
               "Bernard", "Cafe", "Etoile", "Rivage", "Moulin", "Fleur"],
}
K = {
    "US": ["Plumbing", "Logistics", "Bakery", "Dental", "Motors", "Consulting", "Hardware", "Realty"],
    "India": ["Traders", "Textiles", "Engineering", "Sweets", "Pharma", "Electricals", "Foods", "Infotech"],
    "France": ["Boulangerie", "Menuiserie", "Traiteur", "Pharmacie", "Garage", "Librairie", "Fromagerie", "Plomberie"],
}
SUF = {"US": ["Inc", "LLC", "Corp", "Co"], "India": ["Pvt Ltd", "Private Limited", "LLP"],
       "France": ["SARL", "SAS", "SA"]}
CITY = {
    "US": [("Springfield", "IL", "627"), ("Austin", "TX", "787"), ("Denver", "CO", "802"), ("Boston", "MA", "021")],
    "India": [("Pune", "Maharashtra", "411"), ("Mumbai", "Maharashtra", "400"), ("Jaipur", "Rajasthan", "302"),
              ("Chennai", "Tamil Nadu", "600")],
    "France": [("Paris", "", "750"), ("Lyon", "", "690"), ("Lille", "", "590"), ("Bordeaux", "", "330")],
}
STREET = {
    "US": ["Main Street", "Oak Avenue", "Park Road", "Market Street", "Elm Boulevard"],
    "India": ["MG Road", "Station Road", "Gandhi Nagar", "Nehru Street", "Tilak Road"],
    "France": ["rue de la Paix", "avenue Victor Hugo", "boulevard Saint-Germain", "rue des Lilas", "place Bellecour"],
}
LAND = ["Near SBI ATM", "Opp. City Mall", "Behind Bus Stand"]


def typo(s, r):
    if len(s) < 5 or r.random() > 0.5:
        return s
    i = r.randrange(1, len(s) - 2)
    return r.choice([s[:i] + s[i + 1:], s[:i] + s[i + 1] + s[i] + s[i + 2:]])


def make_entity(r, c, used):
    while True:
        core = f"{r.choice(W[c])} {r.choice(W[c])} {r.choice(K[c])}" if r.random() < .5 else f"{r.choice(W[c])} {r.choice(K[c])}"
        if core not in used:
            used.add(core)
            break
    suf = r.choice(SUF[c])
    city, state, pre = r.choice(CITY[c])
    pc = pre + f"{r.randrange(100):03d}"[: 2 if c != "India" else 3]
    pc = (pre + f"{r.randrange(1000):03d}")[:6 if c == "India" else 5]
    no = r.randrange(1, 300)
    return dict(core=core, suf=suf, city=city, state=state, pc=pc, no=no, street=r.choice(STREET[c]),
                land=r.choice(LAND), c=c)


def render(e, r, noise):
    core, suf = e["core"], e["suf"]
    if noise:
        if r.random() < .3:
            core = " ".join(reversed(core.split())) if r.random() < .3 else core
        core = " ".join(typo(w, r) for w in core.split()) if r.random() < .4 else core
        if r.random() < .3:
            core = core.replace(" and ", " & ")
        suf = r.choice(["", suf, suf.replace("Private Limited", "Pvt Ltd").replace("Corp", "Corporation")
                        .replace("Inc", "Incorporated")])
        if r.random() < .2:
            core = core.upper()
    name = f"{core} {suf}".strip()
    st = e["street"]
    if noise:
        st = st.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave").replace("rue", "Rue")
        if r.random() < .2:
            st = st.replace("é", "e")
    parts = [f"{e['no']} {st}" if e["c"] != "India" else f"Plot {e['no']}, {st}", e["city"], e["state"]]
    if e["c"] == "France":
        parts = [f"{e['no']} {st}", f"{e['pc']} {e['city']}"]
    elif not (noise and r.random() < .3):
        parts.append(e["pc"])
    if noise and e["c"] == "India" and r.random() < .3:
        parts.insert(1, e["land"])
    if noise and r.random() < .15:
        parts = parts[1:]
    addr = ", ".join(p for p in parts if p)
    return name, addr


def gen(seed, countries, n_per, out_dir, prefix, truth_path):
    r = random.Random(seed)
    used, ents = set(), []
    for c in countries:
        for _ in range(n_per):
            ents.append(make_entity(r, c, used))
    s1, s2, s3, truth = [], [], [], {}
    n2 = n3 = 0
    for k, e in enumerate(ents, 1):
        sid = f"S1-{k:05d}"
        nm, ad = render(e, r, False)
        s1.append((sid, nm, ad, e["c"]))
        truth[sid] = []
        for _ in range(r.choice([0, 1, 1, 2])):  # 0..2 records in each of S2/S3
            if r.random() < .55:
                n2 += 1; i2 = f"S2-{n2:05d}"
                s2.append((i2,) + render(e, r, True) + (e["c"],)); truth[sid].append(i2)
            if r.random() < .5:
                n3 += 1; i3 = f"S3-{n3:05d}"
                s3.append((i3,) + render(e, r, True) + (e["c"],)); truth[sid].append(i3)
    for c in countries:  # distractors that belong to no S1 entity
        for _ in range(n_per // 3):
            e = make_entity(r, c, used)
            if r.random() < .5:
                n2 += 1; s2.append((f"S2-{n2:05d}",) + render(e, r, True) + (c,))
            else:
                n3 += 1; s3.append((f"S3-{n3:05d}",) + render(e, r, True) + (c,))
    os.makedirs(out_dir, exist_ok=True)
    for k, rows in ((1, s1), (2, s2), (3, s3)):
        r.shuffle(rows) if k > 1 else None
        with open(os.path.join(out_dir, f"{prefix}_source{k}.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for row in rows:
                f.write("\t".join(row) + "\n")
    with open(truth_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s, m in truth.items():
            f.write(s + "\t" + ",".join(m) + "\n")


if __name__ == "__main__":
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    gen(1, ["US", "India"], n, f"{root}/dataset/train", "train", f"{root}/dataset/train/train_ground_truth.tsv")
    gen(2, ["US", "India", "France"], n // 2, f"{root}/dataset/test", "test", f"{root}/hidden_test_truth.tsv")
    print("synthetic data written to", root)
