# Skyline

A skyline is set of points which is not dominated by any other point.

This implementation uses a block nested loop. Identical observations are all part of the skyline if applicable.

## Parameters

- **minimize**

  *Type* → `list[base.typing.FeatureName] | None`

  *Default* → `None`

  A list of features for which the values need to be minimized. Can be omitted as long as `maximize` is specified.
- **maximize**

  *Type* → `list[base.typing.FeatureName] | None`

  *Default* → `None`

  A list of features for which the values need to be maximized. Can be omitted as long as `minimize` is specified.

## Examples

Here is an example taken from [this](<https://maxhalford.github.io/blog/skyline-queries/>) blog post.

```python
import random
from river import misc

city_prices = {
    'Bordeaux': 4045,
    'Lyon': 4547,
    'Toulouse': 3278
}

def random_house():
    city = random.choice(['Bordeaux', 'Lyon', 'Toulouse'])
    size = round(random.gauss(200, 50))
    price = round(random.uniform(0.8, 1.2) * city_prices[city] * size)
    return {'city': city, 'size': size, 'price': price}

skyline = misc.Skyline(minimize=['price'], maximize=['size'])

random.seed(42)

for _ in range(100):
    house = random_house()
    skyline.update(house)

print(len(skyline))
```

```text
13
```

```python
print(skyline[0])
```

```text
{'city': 'Toulouse', 'size': 280, 'price': 763202}
```

Here is another example using the kart data from *Mario Kart: Double Dash!!*.

```python
import collections
from river import misc

Kart = collections.namedtuple(
     'Kart',
     'name speed off_road acceleration weight turbo'
)

karts = [
    Kart('Red Fire', 5, 4, 4, 5, 2),
    Kart('Green Fire', 7, 3, 3, 4, 2),
    Kart('Heart Coach', 4, 6, 6, 5, 2),
    Kart('Bloom Coach', 6, 4, 5, 3, 2),
    Kart('Turbo Yoshi', 4, 5, 6, 6, 2),
    Kart('Turbo Birdo', 6, 4, 4, 7, 2),
    Kart('Goo-Goo Buggy', 1, 9, 9, 2, 3),
    Kart('Rattle Buggy', 2, 9, 8, 2, 3),
    Kart('Toad Kart', 3, 9, 7, 2, 3),
    Kart('Toadette Kart', 1, 9, 9, 2, 3),
    Kart('Koopa Dasher', 2, 8, 8, 3, 3),
    Kart('Para-Wing', 1, 8, 9, 3, 3),
    Kart('DK Jumbo', 8, 2, 2, 8, 1),
    Kart('Barrel Train', 8, 7, 3, 5, 3),
    Kart('Koopa King', 9, 1, 1, 9, 1),
    Kart('Bullet Blaster', 8, 1, 4, 1, 3),
    Kart('Wario Car', 7, 3, 3, 7, 1),
    Kart('Waluigi Racer', 5, 9, 5, 6, 2),
    Kart('Piranha Pipes', 8, 7, 2, 9, 1),
    Kart('Boo Pipes', 2, 9, 8, 9, 1),
    Kart('Parade Kart', 7, 3, 4, 7, 3)
]

skyline = misc.Skyline(
    maximize=['speed', 'off_road', 'acceleration', 'turbo'],
    minimize=['weight']
)

for kart in karts:
    skyline.update(kart._asdict())

best_cart_names = [kart['name'] for kart in skyline]
for name in best_cart_names:
    print(f'- {name}')
```

```text
- Green Fire
- Heart Coach
- Bloom Coach
- Goo-Goo Buggy
- Rattle Buggy
- Toad Kart
- Toadette Kart
- Barrel Train
- Koopa King
- Bullet Blaster
- Waluigi Racer
- Parade Kart
```

```python
for name in sorted(set(kart.name for kart in karts) - set(best_cart_names)):
    print(f'- {name}')
```

```text
- Boo Pipes
- DK Jumbo
- Koopa Dasher
- Para-Wing
- Piranha Pipes
- Red Fire
- Turbo Birdo
- Turbo Yoshi
- Wario Car
```

## Methods

## References

- [Skyline queries in Python](<https://maxhalford.github.io/blog/skyline-queries-in-python>)
- [Borzsony, S., Kossmann, D. and Stocker, K., 2001, April. The skyline operator. In Proceedings 17th international conference on data engineering (pp. 421-430). IEEE.](<https://infolab.usc.edu/csci599/Fall2007/papers/e-1.pdf>)
- [Tao, Y. and Papadias, D., 2006. Maintaining sliding window skylines on data streams. IEEE Transactions on Knowledge and Data Engineering, 18(3), pp.377-391.](<http://www.cs.ust.hk/~dimitris/PAPERS/TKDE06-Sky.pdf>)
