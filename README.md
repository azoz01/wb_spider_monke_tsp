# Discrete Spider monke optimization for TSP problem
#### [Implementation of discrete spider monkey optimization for traveling salesman problem](https://www.youtube.com/watch?v=1XwimHTXYdQ).

<p align="center">
    <img src="https://user-images.githubusercontent.com/44975359/236620465-8a519905-4af2-44e9-8c12-ba91c26f6860.png" width="250" height="250" style="display: block; margin: 0 auto"/>
</p>


## Usage
Run optimization
```
python monke.py PROBLEM_PATH --config-path=config/optmizer_config.yaml --n-iter=10000 --timeout-seconds=300
```

## Help
```
Usage: python monke.py [OPTIONS] PROBLEM_PATH

  Optimize traveling salesman problem using spider-monkey discrete optimization algorithm.

Arguments:
  PROBLEM_PATH  Path to problem.  [required]

Options:
  --config-path TEXT              Path to configuration of optimizer. [default: config/optmizer_config.yaml]
  --n-iter INTEGER                Number of iterations to perform.  [default: 10000]
  --timeout-seconds INTEGER       Maximum optimization time  [default: 300]
```

## Sources
* [Algorithm](https://www.sciencedirect.com/science/article/pii/S1568494619306684)
* [Data](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/)
* [Known solutions](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ATSP.html)

```
ftv38: 1779
[24, 36, 14, 16,  1, 29, 31, 32, 34, 38,  2,  3,  0, 18, 17, 19, 28,
       27, 22, 21, 20, 13, 10, 12, 11, 37,  9,  8, 15,  5,  7,  6,  4, 35,
       33, 30, 25, 26, 23]
       
ft70: 46864
[10, 43, 34, 32, 12,  8, 39, 69, 64, 48, 16, 15,  2, 56, 53,  4, 22,
       14, 63, 66, 29, 23, 30, 37, 60, 59, 57, 44, 27, 24,  5,  9,  7, 13,
       20, 18, 41, 40, 36, 51, 33, 31, 19, 47, 55, 50, 17, 25,  3, 38, 35,
       42, 28, 54, 52, 21, 26, 46, 45, 49, 67, 65, 68, 62, 61, 58,  6,  0,
        1, 11]
        

rgb405: 4371
[  4,  39, 139,  30,  24, 294, 292, 145, 244, 344, 133, 398, 111,
       242, 389, 186, 340, 170, 378,  18, 313, 149, 173, 235, 362,  85,
        26, 223, 167,  46, 332, 355, 373,   0,  66,  74, 115, 102,  35,
       117, 221,  34,  21, 402,  86, 200,  38, 257, 236,  43, 298, 192,
       338, 196, 129, 250, 258, 103, 176,  31, 254, 208, 199, 382,  19,
       142, 229, 393, 151,  68, 124, 238, 303,  17, 352,  91,  94,  79,
       263,  99,  67, 336, 157,  78, 288, 280,  32, 156, 147,  92, 347,
       318,  89, 359,  28, 101,  55, 297, 397,  77, 348, 277, 251,  54,
       314, 185,  84, 197, 215, 206, 188,   7, 158, 337, 209, 181, 323,
        64,  25, 141, 278, 198, 333, 114,  97, 394, 233, 329,   5, 249,
       264, 138, 122, 367, 310, 283, 161, 400, 266, 207, 341,  87, 260,
       137,  90, 363, 282, 281, 308, 289,  95, 396, 327, 131, 368, 160,
       261, 153,  76,  40, 150, 346, 385, 300,  96, 159,  93, 349, 302,
        50, 202, 216, 345, 273, 322, 272, 175, 194, 380, 224,  82,  13,
       205, 132, 334, 193,  27, 116,  81, 276, 259, 374, 148, 255, 306,
       168, 109, 110,  63, 265, 180,   1, 377,  58,   8, 171, 312,  56,
       253, 342, 126, 335, 146,  12, 213, 189, 155, 165,  23,  52, 108,
       381, 211, 350, 130, 245, 248, 275, 351, 326, 339,  29, 166, 379,
       234, 366, 321,  61, 107, 226, 204, 172,  10, 220,  11,  20, 252,
       191, 287, 121, 246,  51, 225, 177, 231, 325,  65, 262,  14, 136,
       330, 247, 120,  60, 391, 299,  49, 104, 227, 212, 134, 320, 375,
       305,  80, 184, 399,   6, 239,  70, 135, 324,  16,   2, 364,  72,
        71, 169, 269, 217, 384,  15,  57, 162, 316, 113, 386, 395, 304,
       295, 361, 237, 163, 365, 187, 307,  75, 271,  83, 183, 123,  69,
       284, 343, 230, 290, 164,  22, 119, 214, 268, 279, 174, 241, 317,
       195, 370, 270, 144, 387, 267, 301, 286, 118, 383, 203,   9, 372,
        41,   3, 357, 358, 371, 293,  42, 401, 319,  33, 201, 128, 210,
       222,  88,  62, 291,  36, 105, 390, 182, 360, 353, 388,  73, 179,
       240, 243, 256, 331, 376, 309, 218,  98, 140, 143, 392, 228, 369,
        53, 354,  59, 296, 315, 106, 190, 178, 154, 232, 125,  37, 127,
       285, 311, 356, 100,  44,  48, 274, 112, 328, 219,  47,  45, 152]
```
