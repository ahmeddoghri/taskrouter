import json, math, random

EXPERTS=[lambda x:2*x+1, lambda x:x*x-.5, lambda x:math.sin(x)*3]
PROTOS=[-2.0,0.0,2.0]

def run(seed=23):
    rng=random.Random(seed); static=route=0
    for _ in range(240):
        task=rng.randrange(3); x=PROTOS[task]+rng.uniform(-.45,.45)
        y=EXPERTS[task](x)
        merged=sum(f(x) for f in EXPERTS)/len(EXPERTS)
        chosen=min(range(3), key=lambda k:abs(x-PROTOS[k]))
        static+=abs(merged-y); route+=abs(EXPERTS[chosen](x)-y)
    return {"static_merge_mae":round(static/240,3),"routed_mae":round(route/240,3),
            "error_reduction_pct":round(100*(1-route/static),1)}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
