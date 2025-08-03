#define _USE_MATH_DEFINES

#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Communicator.h"
#include "tools/Vector.h"
#include "tools/Log.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

/*******************************************************************************
 * Level Surface (PLUMED CUSTOM CV)
 *
 * A(ρ0) = ∫δ(ρ(r) - ρ0) |∇ρ(r)| dV
 * = Σδ(ρ(r) - ρ0) |∇ρ(r)| ΔV
 *
 * surf: LEVEL_SURFACE ATOMS=300-700 LEVEL=1.7 GRIDSIZE=32 SIGMA=0.68 DELTASIGMA=0.1
 *
 * Calculate the surface area of a given level set within a 3D density function using coarea formula.
 * For more information:
 * https://en.wikipedia.org/wiki/Coarea_formula
 *
 * Author: Doyoung Hong ( tox1018@gmail.com )
 * Date: Jul 31, 2025
 * Tested on: PLUMED 2.8.3
 *******************************************************************************/

inline double smoothDelta(double x, double sigma) //δ(x)
{
    double a = x / sigma;
    double norm = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);
    return norm * std::exp(-0.5 * a * a);
}

inline double dsmoothDelta_dx(double x, double sigma)  //dδ(x)/dx
{
    double val = smoothDelta(x, sigma);
    return -(x / (sigma * sigma)) * val;
}

namespace PLMD
{
    namespace colvar
    {
        struct Grid3D
        {
            int nx, ny, nz;
            double dx, dy, dz;
            std::vector<double> rho;
            std::vector<double> gradx;
            std::vector<double> grady;
            std::vector<double> gradz;
        };

        class LevelSurface : public Colvar
        {
        private:
            int nx, ny, nz;
            double level;
            double deltaSigma;
            double sigma;
            unsigned NumParallel_; //number of parallel tasks
            std::vector<size_t> myAtoms;
            int attempts = 0;
        public:
            static void registerKeywords(Keywords& keys);
            explicit LevelSurface(const ActionOptions& ao);
            void computeDensityAndGradient(const std::vector<Vector>& atomPositions, double boxL, Grid3D& grid);
            double computeCoareaSurface(const Grid3D& grid);
            void computeCoareaDerivatives(const Grid3D& grid, const std::vector<Vector>& atomPositions, double boxL, std::vector<Vector>& derivatives);
            void calculate() override;
        };

        PLUMED_REGISTER_ACTION(LevelSurface, "LEVEL_SURFACE");

        void LevelSurface::registerKeywords(Keywords& keys)
        {
            Colvar::registerKeywords(keys);
            keys.add("atoms", "ATOMS", "Atoms to build density function");
            keys.add("compulsory", "LEVEL", "Target density value (e.g. 1.8)");
            keys.add("optional", "GRIDSIZE", "Grid resolution (default=64)");
            keys.add("optional", "SIGMA", "Gaussian width (default=0.68)");
            keys.add("optional", "DELTASIGMA", "Width for smooth delta (default=0.1)");
            keys.addFlag("SERIAL", false, "perform the calculation in serial even if multiple tasks are available");
        }

        LevelSurface::LevelSurface(const ActionOptions& ao) : PLUMED_COLVAR_INIT(ao)
        {
            std::vector<AtomNumber> atoms;
            parseAtomList("ATOMS", atoms);
            parse("LEVEL", level);
            nx = ny = nz = 64;
            parse("GRIDSIZE", nx);
            ny = nx, nz = nx;
            sigma = 0.68;
            parse("SIGMA", sigma);
            deltaSigma = 0.1;
            parse("DELTASIGMA", deltaSigma);
            addValueWithDerivatives();
            setNotPeriodic();
            requestAtoms(atoms);

            NumParallel_ = comm.Get_size();
            unsigned rank = comm.Get_rank();
            bool serial = false;
            parseFlag("SERIAL", serial);
            if (serial)
            {
                log.printf(" -- SERIAL: running without loop parallelization\n");
                NumParallel_ = 1;
                rank = 0;
            }
            size_t totalAtoms = getNumberOfAtoms();
            myAtoms.clear();
            myAtoms.reserve(totalAtoms / NumParallel_ + 1);
            for (size_t i = 0; i < totalAtoms; i++) {
                if (i % NumParallel_ == rank) myAtoms.push_back(i);
            }

            checkRead();
        }

        void LevelSurface::computeDensityAndGradient(const std::vector<Vector>& atomPositions, double boxL, Grid3D& grid)
        {
            int nx = grid.nx, ny = grid.ny, nz = grid.nz;
            double dx = grid.dx, dy = grid.dy, dz = grid.dz;
            std::fill(grid.rho.begin(), grid.rho.end(), 0.0);
            std::fill(grid.gradx.begin(), grid.gradx.end(), 0.0);
            std::fill(grid.grady.begin(), grid.grady.end(), 0.0);
            std::fill(grid.gradz.begin(), grid.gradz.end(), 0.0);
            double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
            double cutoff = 3.0 * sigma;
            double cutoffSq = cutoff * cutoff;
            for (size_t idx = 0; idx < myAtoms.size(); idx++)
            {
                size_t a = myAtoms[idx];

                //Centralize
                double ax = atomPositions[a][0] - boxL / 2.0;
                double ay = atomPositions[a][1] - boxL / 2.0;
                double az = atomPositions[a][2] - boxL / 2.0;

                int ixmin = std::max(0, (int)std::floor((ax - cutoff + boxL / 2.0) / dx));
                int ixmax = std::min(nx - 1, (int)std::ceil((ax + cutoff + boxL / 2.0) / dx));
                int iymin = std::max(0, (int)std::floor((ay - cutoff + boxL / 2.0) / dy));
                int iymax = std::min(ny - 1, (int)std::ceil((ay + cutoff + boxL / 2.0) / dy));
                int izmin = std::max(0, (int)std::floor((az - cutoff + boxL / 2.0) / dz));
                int izmax = std::min(nz - 1, (int)std::ceil((az + cutoff + boxL / 2.0) / dz));

                for (int ix = ixmin; ix <= ixmax; ix++)
                {
                    double gx_ = ix * dx - boxL / 2.0;
                    double dx_ = gx_ - ax;
                    for (int iy = iymin; iy <= iymax; iy++)
                    {
                        double gy_ = iy * dy - boxL / 2.0;
                        double dy_ = gy_ - ay;
                        for (int iz = izmin; iz <= izmax; iz++)
                        {
                            double gz_ = iz * dz - boxL / 2.0;
                            double dz_ = gz_ - az;
                            double r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                            if (r2 < cutoffSq)
                            {
                                double e = std::exp(-r2 * inv2sig2); // = exp(-r^2 / (2*sigma^2))
                                double val = e;
                                double f = -2.0 * inv2sig2; // f = -1/σ^2
                                double grx = f * dx_ * val; // ∂e/∂x = (-x / σ^2) * e
                                double gry = f * dy_ * val; // ∂e/∂y = (-y / σ^2) * e
                                double grz = f * dz_ * val; // ∂e/∂z = (-z / σ^2) * e
                                size_t grid_idx = (size_t)(ix * (ny * nz) + iy * nz + iz); //index3D -> index1D
                                grid.rho[grid_idx] += val; //ρ = Σe (density) 
                                grid.gradx[grid_idx] += grx; //∂ρ/∂x = Σ∂e/∂x
                                grid.grady[grid_idx] += gry; //∂ρ/∂y = Σ∂e/∂y
                                grid.gradz[grid_idx] += grz;//∂ρ/∂z = Σ∂e/∂z
                            }
                        }
                    }
                }
            }
        }

        double LevelSurface::computeCoareaSurface(const Grid3D& grid)
        {
            double surfaceArea = 0.0;
            int nx = grid.nx, ny = grid.ny, nz = grid.nz;
            double cellV = grid.dx * grid.dy * grid.dz;
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    for (int iz = 0; iz < nz; iz++)
                    {
                        size_t idx = (size_t)(ix * (ny * nz) + iy * nz + iz); //index3D -> index1D
                        double r = grid.rho[idx]; //ρ(density)
                        double gx = grid.gradx[idx]; //∂ρ/∂x
                        double gy = grid.grady[idx]; //∂ρ/∂y
                        double gz = grid.gradz[idx]; //∂ρ/∂z
                        double gradmag = std::sqrt(gx * gx + gy * gy + gz * gz); // |∇ρ|
                        double dval = smoothDelta(r - level, deltaSigma); // Only values near rho0 contribute

                        //Coarea formular : Σδ(ρ-ρ0)*|∇ρ|*dV
                        surfaceArea += dval * gradmag * cellV;
                    }
                }
            }
            return surfaceArea;
        }

        void LevelSurface::computeCoareaDerivatives(const Grid3D& grid, const std::vector<Vector>& atomPositions, double boxL, std::vector<Vector>& derivatives)
        {
            for (size_t i = 0; i < derivatives.size(); i++)
            {
                derivatives[i][0] = 0.0;
                derivatives[i][1] = 0.0;
                derivatives[i][2] = 0.0;
            }
            int nx = grid.nx, ny = grid.ny, nz = grid.nz;
            double dx = grid.dx, dy = grid.dy, dz = grid.dz;
            double cellV = dx * dy * dz;
            double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
            double cutoff = 3.0 * sigma;
            double cutoffSq = cutoff * cutoff;
            for (size_t idx = 0; idx < myAtoms.size(); idx++)
            {
                size_t a = myAtoms[idx];
                //Centralize
                double ax = atomPositions[a][0] - boxL / 2.0;
                double ay = atomPositions[a][1] - boxL / 2.0;
                double az = atomPositions[a][2] - boxL / 2.0;

                int ixmin = std::max(0, (int)std::floor((ax - cutoff + boxL / 2.0) / dx));
                int ixmax = std::min(nx - 1, (int)std::ceil((ax + cutoff + boxL / 2.0) / dx));
                int iymin = std::max(0, (int)std::floor((ay - cutoff + boxL / 2.0) / dy));
                int iymax = std::min(ny - 1, (int)std::ceil((ay + cutoff + boxL / 2.0) / dy));
                int izmin = std::max(0, (int)std::floor((az - cutoff + boxL / 2.0) / dz));
                int izmax = std::min(nz - 1, (int)std::ceil((az + cutoff + boxL / 2.0) / dz));

                double dSx = 0.0, dSy = 0.0, dSz = 0.0;
                for (int ix = ixmin; ix <= ixmax; ix++)
                {
                    double gx_ = ix * dx - boxL / 2.0;
                    double dx_ = gx_ - ax;
                    for (int iy = iymin; iy <= iymax; iy++)
                    {
                        double gy_ = iy * dy - boxL / 2.0;
                        double dy_ = gy_ - ay;
                        for (int iz = izmin; iz <= izmax; iz++)
                        {
                            double gz_ = iz * dz - boxL / 2.0;
                            double dz_ = gz_ - az;
                            double r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                            if (r2 < cutoffSq)
                            {
                                size_t grid_idx = (size_t)(ix * (ny * nz) + iy * nz + iz);
                                double r = grid.rho[grid_idx];
                                double gxv = grid.gradx[grid_idx];
                                double gyv = grid.grady[grid_idx];
                                double gzv = grid.gradz[grid_idx];
                                double gradmag = std::sqrt(gxv * gxv + gyv * gyv + gzv * gzv);

                                double deltaDer = dsmoothDelta_dx(r - level, deltaSigma);//δ'(ρ - ρ0)
                                double e = std::exp(-r2 * inv2sig2);
                                double f = -2.0 * inv2sig2;
                                double grxAtom = f * dx_ * e; // ∂e/∂x = (-Δx/σ^2) * exp(...)
                                double gryAtom = f * dy_ * e; // ∂e/∂y = (-Δy/σ^2) * exp(...)
                                double grzAtom = f * dz_ * e; // ∂e/∂z = (-Δz/σ^2) * exp(...)

                                //term1 : δ'(ρ - ρ0)*|∇ρ|*∂e/∂x : primary contribution
                                double term1x = deltaDer * gradmag * (-grxAtom);
                                double term1y = deltaDer * gradmag * (-gryAtom);
                                double term1z = deltaDer * gradmag * (-grzAtom);

                                //term2 : δ(ρ - ρ0) * (∂|∇ρ|/∂x) : secondary contribution (ignored)

                                //dS ? Σδ'(ρ - ρ0)*|∇ρ|*∂e/∂x * dV
                                dSx += term1x * cellV;
                                dSy += term1y * cellV;
                                dSz += term1z * cellV;
                            }
                        }
                    }
                }
                derivatives[a][0] += dSx;
                derivatives[a][1] += dSy;
                derivatives[a][2] += dSz;
            }
        }

        void LevelSurface::calculate()
        {
            using namespace std;
            Tensor box = getBox();
            double boxL = box[0][0];
            if ((box[1][1] != boxL) || (box[2][2] != boxL)) error("LEVEL_SURFACE requires a cubic box!");
            auto positions = getPositions();
            Grid3D grid;
            grid.nx = nx;
            grid.ny = ny;
            grid.nz = nz;
            grid.dx = boxL / (nx - 1);
            grid.dy = boxL / (ny - 1);
            grid.dz = boxL / (nz - 1);
            grid.rho.resize(nx * ny * nz, 0.0);
            grid.gradx.resize(nx * ny * nz, 0.0);
            grid.grady.resize(nx * ny * nz, 0.0);
            grid.gradz.resize(nx * ny * nz, 0.0);

            //auto timeStamp1 = std::chrono::high_resolution_clock::now();
            computeDensityAndGradient(positions, boxL, grid);
            //auto timeStamp2 = std::chrono::high_resolution_clock::now();
            if (NumParallel_ > 1)
            {
                comm.Sum(grid.rho);
                comm.Sum(grid.gradx);
                comm.Sum(grid.grady);
                comm.Sum(grid.gradz);
            }
            //auto timeStamp3 = std::chrono::high_resolution_clock::now();
            double Svalue = computeCoareaSurface(grid);
            vector<Vector> dSdR(getNumberOfAtoms());
            //auto timeStamp4 = std::chrono::high_resolution_clock::now();
            computeCoareaDerivatives(grid, positions, boxL, dSdR);
            //auto timeStamp5 = std::chrono::high_resolution_clock::now();
            if (NumParallel_ > 1)
            {
                std::vector<double> flat_derivatives;
                flat_derivatives.reserve(getNumberOfAtoms() * 3);
                for (size_t i = 0; i < getNumberOfAtoms(); i++)
                {
                    flat_derivatives.push_back(dSdR[i][0]);
                    flat_derivatives.push_back(dSdR[i][1]);
                    flat_derivatives.push_back(dSdR[i][2]);
                }

                comm.Sum(flat_derivatives);

                for (size_t i = 0; i < getNumberOfAtoms(); i++)
                {
                    dSdR[i][0] = flat_derivatives[i * 3 + 0];
                    dSdR[i][1] = flat_derivatives[i * 3 + 1];
                    dSdR[i][2] = flat_derivatives[i * 3 + 2];
                }
            }
            //auto timeStamp6 = std::chrono::high_resolution_clock::now();
            setValue(Svalue);
            for (size_t i = 0; i < getNumberOfAtoms(); i++) setAtomsDerivatives(i, dSdR[i]);
            setBoxDerivativesNoPbc();
            //auto timeStamp7 = std::chrono::high_resolution_clock::now();

            //size_t rank_ = comm.Get_rank();
            //if (rank_ == 0)
            //{
            //    auto time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStamp2 - timeStamp1).count();
            //    auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStamp3 - timeStamp2).count();
            //    auto time3 = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStamp4 - timeStamp3).count();
            //    auto time4 = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStamp5 - timeStamp4).count();
            //    auto time5 = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStamp6 - timeStamp5).count();
            //    auto time6 = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStamp7 - timeStamp6).count();
            //    printf("Timing - DensityGrid: %ld ns, GridComm: %ld ns, CoareaSurface: %ld ns, Derivatives: %ld ns, DerivComm: %ld ns, SetResults: %ld ns\n",
            //        time1, time2, time3, time4, time5, time6);
            //    attempts++;
            //}
            //if (attempts > 10) exit(-1);
        }
    }
}
