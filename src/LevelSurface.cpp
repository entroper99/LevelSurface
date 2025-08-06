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
 * Version 0.44
 *
 * A(ρ0) = ∫δ(ρ(r) - ρ0) |∇ρ(r)| dV
 * = Σδ(ρ(r) - ρ0) |∇ρ(r)| ΔV
 *
 * surf: LEVEL_SURFACE ATOMS=300-700 LEVEL=0.3635 GRIDSIZE=32 SIGMA=0.68 DELTASIGMA=0.0202
 *
 * Calculate the surface area of a given level set within a 3D density function using coarea formula.
 * For more information:
 * https://en.wikipedia.org/wiki/Coarea_formula
 *
 * Author: Doyoung Hong ( tox1018@gmail.com )
 * Developed with AI assistance from Claude, Gemini, and GPT
 * Date: Mar 11, 2025
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
            std::vector<double> data;

            inline size_t index(int x, int y, int z) const { return static_cast<size_t>(x * ny * nz + y * nz + z); }
            inline double& rho_at(int x, int y, int z) { return data[index(x, y, z) * 4 + 0]; }
            inline double& gradx_at(int x, int y, int z) { return data[index(x, y, z) * 4 + 1]; }
            inline double& grady_at(int x, int y, int z) { return data[index(x, y, z) * 4 + 2]; }
            inline double& gradz_at(int x, int y, int z) { return data[index(x, y, z) * 4 + 3]; }
        };

        class LevelSurface : public Colvar
        {
        private:
            int nx, ny, nz;
            double level;
            double deltaSigma;
            double sigma;
            Grid3D grid;
            unsigned NumParallel_; //number of parallel tasks
            std::vector<size_t> myAtoms;
            int attempts = 0;
            std::vector<double> derivs;
        public:
            static void registerKeywords(Keywords& keys);
            explicit LevelSurface(const ActionOptions& ao);
            void computeDensityAndGradient(const std::vector<Vector>& atomPositions, double boxL);
            double computeCoareaSurface();
            void computeCoareaDerivatives(const std::vector<Vector>& atomPositions, double boxL);
            void calculate() override;
        };

        PLUMED_REGISTER_ACTION(LevelSurface, "LEVEL_SURFACE");

        void LevelSurface::registerKeywords(Keywords& keys)
        {
            Colvar::registerKeywords(keys);
            keys.add("atoms", "ATOMS", "Atoms to build density function");
            keys.add("compulsory", "LEVEL", "Target density value (e.g. 1.8)");
            keys.add("optional", "GRIDSIZE", "Grid resolution (default=32)");
            keys.add("optional", "SIGMA", "Gaussian width (default=0.68)");
            keys.add("optional", "DELTASIGMA", "Width for smooth delta (default= 0.0202)");
            keys.addFlag("SERIAL", false, "perform the calculation in serial even if multiple tasks are available");
        }

        LevelSurface::LevelSurface(const ActionOptions& ao) : PLUMED_COLVAR_INIT(ao)
        {
            std::vector<AtomNumber> atoms;
            parseAtomList("ATOMS", atoms);
            parse("LEVEL", level);
            nx = ny = nz = 32;
            parse("GRIDSIZE", nx);
            ny = nx, nz = nx;
            sigma = 0.68;
            parse("SIGMA", sigma);
            deltaSigma = 0.0202;
            parse("DELTASIGMA", deltaSigma);
            addValueWithDerivatives();
            setNotPeriodic();
            requestAtoms(atoms);

            grid.data.resize(nx * ny * nz * 4, 0.0);
            derivs.resize(getNumberOfAtoms() * 3, 0.0);

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
            for (size_t i = 0; i < totalAtoms; i++)
            {
                if (i % NumParallel_ == rank) myAtoms.push_back(i);
            }

            checkRead();
        }

        void LevelSurface::computeDensityAndGradient(const std::vector<Vector>& atomPositions, double boxL)
        {
            std::fill(grid.data.begin(), grid.data.end(), 0.0);
            const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
            const double dx = grid.dx, dy = grid.dy, dz = grid.dz;
            const double norm = 1.0 / (std::pow(2.0 * M_PI, 1.5) * sigma * sigma * sigma); // 1 / ( (2π)^(3/2) σ^3 )

            const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
            const double cutoff = 3.0 * sigma;
            const double cutoffSq = cutoff * cutoff;

            const double invdx = 1.0 / dx, invdy = 1.0 / dy, invdz = 1.0 / dz;
            const double cx = cutoff * invdx;
            const double cy = cutoff * invdy;
            const double cz = cutoff * invdz;

            for (size_t idx = 0; idx < myAtoms.size(); idx++)
            {
                const size_t a = myAtoms[idx];

                const double ax = atomPositions[a][0] - boxL * 0.5;
                const double ay = atomPositions[a][1] - boxL * 0.5;
                const double az = atomPositions[a][2] - boxL * 0.5;

                const double ux = (ax + boxL * 0.5) * invdx - 0.5;
                const double uy = (ay + boxL * 0.5) * invdy - 0.5;
                const double uz = (az + boxL * 0.5) * invdz - 0.5;

                const int ixmin = (int)std::floor(ux - cx);
                const int ixmax = (int)std::ceil(ux + cx);
                const int iymin = (int)std::floor(uy - cy);
                const int iymax = (int)std::ceil(uy + cy);
                const int izmin = (int)std::floor(uz - cz);
                const int izmax = (int)std::ceil(uz + cz);

                for (int iz = izmin; iz <= izmax; ++iz)
                {
                    const int kwrap = ((iz % nz) + nz) % nz;
                    const double gz_ = (kwrap + 0.5) * dz - boxL * 0.5;

                    for (int iy = iymin; iy <= iymax; ++iy)
                    {
                        const int jwrap = ((iy % ny) + ny) % ny;
                        const double gy_ = (jwrap + 0.5) * dy - boxL * 0.5;

                        for (int ix = ixmin; ix <= ixmax; ++ix)
                        {
                            const int iwrap = ((ix % nx) + nx) % nx;
                            const double gx_ = (iwrap + 0.5) * dx - boxL * 0.5;

                            double dx_ = gx_ - ax;
                            double dy_ = gy_ - ay;
                            double dz_ = gz_ - az;
                            dx_ -= boxL * std::nearbyint(dx_ / boxL);
                            dy_ -= boxL * std::nearbyint(dy_ / boxL);
                            dz_ -= boxL * std::nearbyint(dz_ / boxL);

                            const double r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                            if (r2 >= cutoffSq) continue;

                            const double eRaw = std::exp(-r2 * inv2sig2); // exp(-r^2/(2σ^2))
                            const double e = norm * eRaw;
                            const double f = -2.0 * inv2sig2;          // = -1/σ^2

                            const double grx = f * dx_ * e;            // ∂e/∂x
                            const double gry = f * dy_ * e;            // ∂e/∂y
                            const double grz = f * dz_ * e;            // ∂e/∂z

                            const size_t grid_idx = (size_t)(iwrap * (ny * nz) + jwrap * nz + kwrap);

                            grid.rho_at(iwrap, jwrap, kwrap) += e;
                            grid.gradx_at(iwrap, jwrap, kwrap) += grx;
                            grid.grady_at(iwrap, jwrap, kwrap) += gry;
                            grid.gradz_at(iwrap, jwrap, kwrap) += grz;
                        }
                    }
                }
            }
        }

        double LevelSurface::computeCoareaSurface()
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
                        double r = grid.rho_at(ix, iy, iz); //ρ(density)
                        double gx = grid.gradx_at(ix, iy, iz); //∂ρ/∂x
                        double gy = grid.grady_at(ix, iy, iz); //∂ρ/∂y
                        double gz = grid.gradz_at(ix, iy, iz); //∂ρ/∂z
                        double gradmag = std::sqrt(gx * gx + gy * gy + gz * gz); // |∇ρ|
                        double dval = smoothDelta(r - level, deltaSigma); // Only values near rho0 contribute

                        //Coarea formular : Σδ(ρ-ρ0)*|∇ρ|*dV
                        surfaceArea += dval * gradmag * cellV;
                    }
                }
            }
            return surfaceArea;
        }

        void LevelSurface::computeCoareaDerivatives(const std::vector<Vector>& atomPositions, double boxL)
        {
            std::fill(derivs.begin(), derivs.end(), 0.0);
            const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
            const double dx = grid.dx, dy = grid.dy, dz = grid.dz;
            const double cellV = dx * dy * dz;
            const double norm = 1.0 / (std::pow(2.0 * M_PI, 1.5) * sigma * sigma * sigma); // 1 / ( (2π)^(3/2) σ^3 )

            const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);   // = 1/(2σ²)
            const double invsig2 = 1.0 / (sigma * sigma);         // = 1/σ²
            const double invsig4 = invsig2 * invsig2;             // = 1/σ⁴

            const double cutoff = 3.0 * sigma;
            const double cutoffSq = cutoff * cutoff;

            const double invdx = 1.0 / dx, invdy = 1.0 / dy, invdz = 1.0 / dz;
            const double cx = cutoff * invdx;
            const double cy = cutoff * invdy;
            const double cz = cutoff * invdz;

            const double eps_n = 1e-12;

            for (size_t idx = 0; idx < myAtoms.size(); idx++)
            {
                const size_t a = myAtoms[idx];

                const double ax = atomPositions[a][0] - boxL * 0.5;
                const double ay = atomPositions[a][1] - boxL * 0.5;
                const double az = atomPositions[a][2] - boxL * 0.5;

                const double ux = (ax + boxL * 0.5) * invdx - 0.5;
                const double uy = (ay + boxL * 0.5) * invdy - 0.5;
                const double uz = (az + boxL * 0.5) * invdz - 0.5;

                const int ixmin = (int)std::floor(ux - cx);
                const int ixmax = (int)std::ceil(ux + cx);
                const int iymin = (int)std::floor(uy - cy);
                const int iymax = (int)std::ceil(uy + cy);
                const int izmin = (int)std::floor(uz - cz);
                const int izmax = (int)std::ceil(uz + cz);

                double dSx = 0.0, dSy = 0.0, dSz = 0.0;

                for (int iz = izmin; iz <= izmax; ++iz) 
                {
                    const int kwrap = ((iz % nz) + nz) % nz;
                    const double gz_ = (kwrap + 0.5) * dz - boxL * 0.5;
                    for (int iy = iymin; iy <= iymax; ++iy) 
                    {
                        const int jwrap = ((iy % ny) + ny) % ny;
                        const double gy_ = (jwrap + 0.5) * dy - boxL * 0.5;
                        for (int ix = ixmin; ix <= ixmax; ++ix) 
                        {
                            const int iwrap = ((ix % nx) + nx) % nx;
                            const double gx_ = (iwrap + 0.5) * dx - boxL * 0.5;

                            double dx_ = gx_ - ax;
                            double dy_ = gy_ - ay;
                            double dz_ = gz_ - az;
                            dx_ -= boxL * std::nearbyint(dx_ / boxL);
                            dy_ -= boxL * std::nearbyint(dy_ / boxL);
                            dz_ -= boxL * std::nearbyint(dz_ / boxL);

                            const double r2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                            if (r2 >= cutoffSq) continue;

                            const size_t grid_idx = (size_t)(iwrap * (ny * nz) + jwrap * nz + kwrap);

                            const double r = grid.rho_at(iwrap, jwrap, kwrap);
                            const double gxv = grid.gradx_at(iwrap, jwrap, kwrap);
                            const double gyv = grid.grady_at(iwrap, jwrap, kwrap);
                            const double gzv = grid.gradz_at(iwrap, jwrap, kwrap);
                            const double gradmag = std::sqrt(gxv * gxv + gyv * gyv + gzv * gzv);

                            const double delta0 = smoothDelta(r - level, deltaSigma);
                            const double deltaDer = dsmoothDelta_dx(r - level, deltaSigma);

                            const double e = norm * std::exp(-r2 * inv2sig2);
                            const double f = -2.0 * inv2sig2; // = -1/σ²
                            const double grxAtom = f * dx_ * e; // ∂e/∂x_atom
                            const double gryAtom = f * dy_ * e; // ∂e/∂y_atom
                            const double grzAtom = f * dz_ * e; // ∂e/∂z_atom

                            // ========= TERM 1 =========
                            // δ'(ρ-ρ0) * |∇ρ| * (-∂e/∂R_a) * dV
                            dSx += (deltaDer * gradmag * (-grxAtom)) * cellV;
                            dSy += (deltaDer * gradmag * (-gryAtom)) * cellV;
                            dSz += (deltaDer * gradmag * (-grzAtom)) * cellV;

                            // ========= TERM 2 =========
                            // δ(ρ-ρ0) * ( ∇ρ / |∇ρ| ) · ∇( ∂ρ/∂R_a )
                            // with ∂ρ/∂R_a = -∂e/∂x etc., so ∇(∂ρ/∂R_a) = -H_:component
                            if (gradmag > eps_n)
                            {
                                const double invg = 1.0 / gradmag;
                                const double nxn = gxv * invg;
                                const double nyn = gyv * invg;
                                const double nzn = gzv * invg;

                                // Hessian of Gaussian e: H_ij = (d_i d_j / σ⁴ - δ_ij / σ²) * e
                                const double Hxx = (dx_ * dx_ * invsig4 - invsig2) * e;
                                const double Hyy = (dy_ * dy_ * invsig4 - invsig2) * e;
                                const double Hzz = (dz_ * dz_ * invsig4 - invsig2) * e;
                                const double Hxy = (dx_ * dy_ * invsig4) * e;
                                const double Hxz = (dx_ * dz_ * invsig4) * e;
                                const double Hyz = (dy_ * dz_ * invsig4) * e;

                                // n · ( -H_:component )
                                const double t2x = -(nxn * Hxx + nyn * Hxy + nzn * Hxz);
                                const double t2y = -(nxn * Hxy + nyn * Hyy + nzn * Hyz);
                                const double t2z = -(nxn * Hxz + nyn * Hyz + nzn * Hzz);

                                dSx += delta0 * t2x * cellV;
                                dSy += delta0 * t2y * cellV;
                                dSz += delta0 * t2z * cellV;
                            }
                        }
                    }
                }

                derivs[a * 3 + 0] += dSx;
                derivs[a * 3 + 1] += dSy;
                derivs[a * 3 + 2] += dSz;
            }
        }


        void LevelSurface::calculate()
        {
            using namespace std;
            Tensor box = getBox();
            double boxL = box[0][0];
            if ((box[1][1] != boxL) || (box[2][2] != boxL)) error("LEVEL_SURFACE requires a cubic box!");
            auto positions = getPositions();

            grid.nx = nx;
            grid.ny = ny;
            grid.nz = nz;
            grid.dx = boxL / nx;
            grid.dy = boxL / ny;
            grid.dz = boxL / nz;


            auto timeStamp1 = std::chrono::high_resolution_clock::now();
            computeDensityAndGradient(positions, boxL);
            auto timeStamp2 = std::chrono::high_resolution_clock::now();
            if (NumParallel_ > 1) comm.Sum(grid.data);
            auto timeStamp3 = std::chrono::high_resolution_clock::now();
            double Svalue = computeCoareaSurface();
            auto timeStamp4 = std::chrono::high_resolution_clock::now();
            computeCoareaDerivatives(positions, boxL);
            auto timeStamp5 = std::chrono::high_resolution_clock::now();
            if (NumParallel_ > 1) comm.Sum(derivs);
            auto timeStamp6 = std::chrono::high_resolution_clock::now();
            setValue(Svalue);
            for (size_t i = 0; i < getNumberOfAtoms(); i++) setAtomsDerivatives(i, { derivs[3 * i + 0],derivs[3 * i + 1],derivs[3 * i + 2] });
            setBoxDerivativesNoPbc();
            auto timeStamp7 = std::chrono::high_resolution_clock::now();

            size_t rank_ = comm.Get_rank();
            if (rank_ == 0)
            {
                using namespace std::chrono;

                double time1 = duration<double, std::milli>(timeStamp2 - timeStamp1).count();
                double time2 = duration<double, std::milli>(timeStamp3 - timeStamp2).count();
                double time3 = duration<double, std::milli>(timeStamp4 - timeStamp3).count();
                double time4 = duration<double, std::milli>(timeStamp5 - timeStamp4).count();
                double time5 = duration<double, std::milli>(timeStamp6 - timeStamp5).count();
                double time6 = duration<double, std::milli>(timeStamp7 - timeStamp6).count();

                std::cout << std::fixed << std::setprecision(3);
                std::cout << "Profile - "
                    << "DensityGrid: " << time1 << " ms, "
                    << "GridComm: " << time2 << " ms, "
                    << "CoareaSurface: " << time3 << " ms, "
                    << "Derivatives: " << time4 << " ms, "
                    << "DerivComm: " << time5 << " ms, "
                    << "SetResults: " << time6 << " ms\n";

                attempts++;
            }
            if (attempts > 10) exit(-1);
        }
    }
}
