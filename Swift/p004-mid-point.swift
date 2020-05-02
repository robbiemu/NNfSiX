/*
Doing dot product with a layer of neurons and multiple inputs
Associated YT NNFS tutorial: https://www.youtube.com/watch?v=tMrbN67U9d4
*/
import Foundation
import Accelerate

extension Array where Element == Double {
    func dot (_ rval: [Double]) -> Double? {
        if self.count != rval.count { return nil }
        return zip(self, rval).reduce(0.0, { (sum, tuple) -> Double in
            sum + tuple.0 * tuple.1
        })
    }

    func add(_ rval: [Double]) -> [Double]? {
        if self.count != rval.count { return nil }
        return zip(self, rval).map(+)
    }
}

extension Array where Element == [Double] {
    func dot (_ rval: [Double]) -> [Double]? {
        let rc = rval.count
        if self.reduce(false, { $1.count != rc }) { return nil }

        return self.map { (lval) -> Double in lval.dot(rval)! }
    }
    
    func dot (_ rval: [[Double]]) -> [[Double]]? {
        if(rval.count != self[0].count) { return nil }

        let rt = rval.T
        return zip(self.indices, self).map{ (index, row) -> [Double] in rt.dot(row)! }
    }
    
    func add(_ rval: [Double]) -> [[Double]]? {
        if self[0].count != rval.count { return nil }
        
        return self.map{ row in zip(row, rval).map(+) }
    }
    
    var T: [[Double]] {
        get {
            var out: [[Double]] = []
            let x = self.count
            let y = self[0].count
            let N = x > y ? x : y
            let M = y < x ? y : x
            for n in 0..<N {
                for m in 0..<M {
                    let a = x > y ? n : m
                    let b = y < x ? m : n
                    if out.count <= b {
                        out.append([])
                    }
                    out[b].append(self[a][b])
                }
            }
            return out
        }
    }
}
// The above replaces numpy =========

let inputs = [[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

let weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]

let biases = [2.0, 3.0, 0.5]

guard let layer1_output:[[Double]] = inputs.dot(weights.T)?.add(biases) else {
    fatalError("Unmatched shapes")
}

let weights2 = [[0.1, -0.14, 0.5],
               [-0.5, 0.12, -0.33],
               [-0.44, 0.73, -0.13]]

let biases2 = [-1.0, 2.0, -0.5]

guard let layer2_output:[[Double]] = layer1_output.dot(weights2.T)?.add(biases2) else {
    fatalError("Unmatched shapes")
}
print(layer2_output)
