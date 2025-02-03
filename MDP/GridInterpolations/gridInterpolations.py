import numpy as np

class AbstractGrid:
    pass

class RectangleGrid(AbstractGrid):
    def __init__(self, *cutPoints):
        self.cutPoints = cutPoints
        self.cut_counts = [len(cutPoints[i]) for i in range(len(cutPoints))]
        self.cuts = np.concatenate(cutPoints)
        
        # Check for duplicates and sorted order
        for i in range(len(cutPoints)):
            if len(set(cutPoints[i])) != len(cutPoints[i]):
                raise ValueError(f"Duplicates cutpoints are not allowed (duplicates observed in dimension {i})")
            if not np.all(np.diff(cutPoints[i]) > 0):
                raise ValueError("Cut points must be sorted")
                
    def __len__(self):
        return np.prod(self.cut_counts)

    def shape(self):
        return tuple(self.cut_counts)

    def dimensions(self):
        return len(self.cutPoints)


class SimplexGrid(RectangleGrid):
    def __init__(self, *cutPoints):
        super().__init__(*cutPoints)
        self.x_p = np.zeros(len(cutPoints))
        self.ihi = np.zeros(len(cutPoints), dtype=int)
        self.ilo = np.zeros(len(cutPoints), dtype=int)
        self.n_ind = np.zeros(len(cutPoints), dtype=int)

    def __len__(self):
        return np.prod(self.cut_counts)

    def shape(self):
        return tuple(self.cut_counts)

    def dimensions(self):
        return len(self.cutPoints)


def ind2x(grid, ind):
    ndims = grid.dimensions()
    x = np.zeros(ndims)
    ind2x_(grid, ind, x)
    return x

def ind2x_(grid, ind, x):
    ndims = grid.dimensions()
    stride = grid.cut_counts[0]
    
    for i in range(1, ndims - 1):
        stride *= grid.cut_counts[i]
    
    for i in range(ndims - 1, -1, -1):
        rest = (ind - 1) % stride + 1
        x[i] = grid.cutPoints[i][(ind - rest) // stride]
        ind = rest
        stride //= grid.cut_counts[i]
    
    x[0] = grid.cutPoints[0][ind]
    
def interpolate(grid, data, x):
    indices, weights = interpolants(grid, x)
    return np.sum(data[indices] * weights)

def maskedInterpolate(grid, data, x, mask):
    indices, weights = interpolants(grid, x)
    val = 0
    totalWeight = 0
    for i in range(len(indices)):
        if mask[indices[i]]:
            continue
        val += data[indices[i]] * weights[i]
        totalWeight += weights[i]
    return val / totalWeight


def interpolants(grid, x):
    if np.any(np.isnan(x)):
        raise ValueError("Input contains NaN!")

    # Assumendo che `RectangleGrid` abbia attributi `cut_counts` e `cuts`
    cut_counts = grid.cut_counts
    cuts = grid.cuts

    # Reset the values in index and weight:
    dimensions = len(cut_counts)
    #num_points = 2 ** dimensions
    num_points = 2**grid.dimensions()
    index = np.zeros(num_points, dtype=int)
    index2 = np.zeros(num_points, dtype=int)
    weight = np.zeros(num_points, dtype=type(x[0]))
    weight2 = np.zeros(num_points, dtype=type(x[0]))

    weight[0] = 1
    weight2[0] = 1

    l = 1
    subblock_size = 1
    cut_i = 0  # Python usa indici basati su 0
    n = 1

    for d in range(len(x)):
        coord = x[d]
        lasti = cut_counts[d] + cut_i - 1
        ii = cut_i

        if coord <= cuts[ii]:
            i_lo, i_hi = ii, ii
        elif coord >= cuts[lasti]:
            i_lo, i_hi = lasti, lasti
        else:
            while cuts[ii] < coord:
                ii += 1
            if cuts[ii] == coord:
                i_lo, i_hi = ii, ii
            else:
                i_lo, i_hi = ii - 1, ii

        # Handle single or double interpolation cases
        if i_lo == i_hi:
            for i in range(l):
                index[i] += (i_lo - cut_i) * subblock_size
        else:
            low = 1 - (coord - cuts[i_lo]) / (cuts[i_hi] - cuts[i_lo])
            for i in range(l):
                index2[i] = index[i] + (i_lo - cut_i) * subblock_size
                index2[i + l] = index[i] + (i_hi - cut_i) * subblock_size
            #index[:2 * l] = index2[:2 * l]
            index[:] = index2
            for i in range(l):
                weight2[i] = weight[i] * low
                weight2[i + l] = weight[i] * (1 - low)
            #weight[:2 * l] = weight2[:2 * l]
            weight[:] = weight2

            l *= 2
            n *= 2

        cut_i += cut_counts[d]
        subblock_size *= cut_counts[d]
        #print(subblock_size)

    v = min(l, len(index))
    #print(l,v)
    return index[:v], weight[:v]


# Now we would use this function to calculate cval (indices and weights for interpolation)
def get_cval(grid, x):
    indices, weights = interpolants(grid, x)
    cval = np.array([indices, weights])  # This is the final cval equivalent in Python
    return cval

def vertices(grid):
    n_dims = grid.dimensions()
    mem = np.zeros((n_dims, len(grid)))

    for idx in range(len(grid)):
        this_idx = idx
        for j in range(n_dims):
            cut_idx = this_idx % grid.cut_counts[j]
            this_idx //= grid.cut_counts[j]
            mem[j, idx] = grid.cutPoints[j][cut_idx]

    return [tuple(mem[:, i]) for i in range(len(grid))]
