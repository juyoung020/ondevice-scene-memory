// 시험 도구: g2o Sim3 자세 그래프(BlockSolver_7_3 + LinearSolverEigen + Levenberg, EdgeSim3 · VertexSim3Expmap)를
// 기준 빌드 원본 라이브러리로 풀어 입력 · 첫 선형화(오차 · 야코비안 · χ² · Hpp · b · x) · 반복마다 정점 값을 덤프한다
// (orb3rs::sim3pg 비트 대조, docs/orb3_port.md "합치기 본질 그래프").
// 실행 파일은 libORB_SLAM3 를 libg2o 보다 먼저 링크해 실제 실행과 같은 기호(Sim3::log 등 약한 기호)를 쓴다.
// 사용: orb3_sim3pg_probe out.bin 그래프수 [씨앗]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#define protected public
#define private public
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/hyper_graph_action.h"
#undef protected
#undef private
#include <System.h>  // libORB_SLAM3 를 실제로 링크하게(약한 기호 순서)

static FILE *F;
template <class T> static void W(const T &v) { fwrite(&v, sizeof(T), 1, F); }
static void Wsim(const g2o::Sim3 &s) {
    W(s.rotation().x()); W(s.rotation().y()); W(s.rotation().z()); W(s.rotation().w());
    for (int i = 0; i < 3; i++) W(s.translation()[i]);
    W(s.scale());
}
struct VDesc { int id; bool fixed, fixscale; g2o::Sim3 est; };
struct EDesc { int a, b; g2o::Sim3 meas; };

static g2o::SparseOptimizer *make(const std::vector<VDesc> &vs, const std::vector<EDesc> &es, g2o::OptimizationAlgorithmLevenberg **alg) {
    auto *opt = new g2o::SparseOptimizer();
    opt->setVerbose(false);
    auto *ls = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    auto *bs = new g2o::BlockSolver_7_3(ls);
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(bs);
    solver->setUserLambdaInit(1e-16);
    opt->setAlgorithm(solver);
    *alg = solver;
    for (auto &d : vs) {
        auto *v = new g2o::VertexSim3Expmap();
        v->setEstimate(d.est);
        v->setFixed(d.fixed);
        v->setId(d.id);
        v->setMarginalized(false);
        v->_fix_scale = d.fixscale;
        opt->addVertex(v);
    }
    const Eigen::Matrix<double, 7, 7> I = Eigen::Matrix<double, 7, 7>::Identity();
    for (auto &d : es) {
        auto *e = new g2o::EdgeSim3();
        e->setVertex(1, opt->vertex(vs[d.b].id));
        e->setVertex(0, opt->vertex(vs[d.a].id));
        e->setMeasurement(d.meas);
        e->information() = I;
        opt->addEdge(e);
    }
    return opt;
}

struct Rec : g2o::HyperGraphAction {
    g2o::SparseOptimizer *opt; std::vector<int> ids; std::vector<std::vector<double>> it;
    HyperGraphAction *operator()(const g2o::HyperGraph *, Parameters *) override {
        std::vector<double> r;
        for (int id : ids) {
            auto *v = static_cast<g2o::VertexSim3Expmap *>(opt->vertex(id));
            const g2o::Sim3 &s = v->estimate();
            r.insert(r.end(), {s.rotation().x(), s.rotation().y(), s.rotation().z(), s.rotation().w(), s.translation()[0], s.translation()[1], s.translation()[2], s.scale()});
        }
        it.push_back(r);
        return this;
    }
};

int main(int argc, char **argv) {
    F = fopen(argv[1], "wb");
    int ng = atoi(argv[2]);
    std::mt19937_64 rng(argc > 3 ? atoll(argv[3]) : 1);
    std::uniform_real_distribution<double> U(-1, 1);
    auto rq = [&]() { Eigen::Quaterniond q(U(rng), U(rng), U(rng), U(rng)); q.normalize(); return q; };
    auto small = [&](double s, bool sc) { Eigen::Matrix<double, 7, 1> u; for (int i = 0; i < 7; i++) u[i] = s * U(rng); if (!sc) u[6] = 0; return g2o::Sim3(u); };
    for (int g = 0; g < ng; g++) {
        int n = 3 + rng() % 40;
        std::vector<VDesc> vs; std::vector<g2o::Sim3> truth;
        int id = rng() % 5;
        int nfix = 1 + rng() % std::max(1, n / 3);
        double noise = (g % 3 == 0) ? 1e-3 : (g % 3 == 1 ? 0.05 : 0.3);
        for (int i = 0; i < n; i++) {
            g2o::Sim3 T(rq(), Eigen::Vector3d(3 * U(rng), 3 * U(rng), 3 * U(rng)), 1.0);
            truth.push_back(T);
            bool fixed = i < nfix;
            g2o::Sim3 est = fixed ? T : small(noise, false) * T;
            vs.push_back({id, fixed, fixed && (rng() % 2 == 0), est});
            id += 1 + rng() % 3;
        }
        std::vector<EDesc> es;
        for (int i = 1; i < n; i++) { int j = rng() % i; es.push_back({i, j, small(noise * 0.3, true) * truth[j] * truth[i].inverse()}); }
        int extra = rng() % (2 * n);
        for (int k = 0; k < extra; k++) { int i = rng() % n, j = rng() % n; if (i == j) continue; es.push_back({i, j, small(noise * 0.3, true) * truth[j] * truth[i].inverse()}); }
        W<uint32_t>(0x31524750); W<int32_t>(n);
        for (auto &d : vs) { W<int32_t>(d.id); W<uint8_t>(d.fixed); W<uint8_t>(d.fixscale); Wsim(d.est); }
        W<int32_t>(es.size());
        for (auto &d : es) { W<int32_t>(d.a); W<int32_t>(d.b); Wsim(d.meas); }
        // 첫 선형화
        {
            g2o::OptimizationAlgorithmLevenberg *alg;
            auto *opt = make(vs, es, &alg);
            opt->initializeOptimization();
            auto *bs = static_cast<g2o::BlockSolver_7_3 *>(alg->solver());
            alg->init();
            bs->buildStructure();
            opt->computeActiveErrors();
            W(opt->activeRobustChi2());
            bs->buildSystem();
            W<int32_t>(opt->activeEdges().size());
            for (auto *e0 : opt->activeEdges()) {
                auto *e = static_cast<g2o::EdgeSim3 *>(e0);
                for (int i = 0; i < 7; i++) W(e->error()[i]);
                W(e->chi2());
                bool fi = static_cast<g2o::OptimizableGraph::Vertex *>(e->vertex(0))->fixed(), fj = static_cast<g2o::OptimizableGraph::Vertex *>(e->vertex(1))->fixed();
                W<uint8_t>(fi); W<uint8_t>(fj);
                for (int i = 0; i < 49; i++) W(fi ? 0.0 : e->jacobianOplusXi().data()[i]);
                for (int i = 0; i < 49; i++) W(fj ? 0.0 : e->jacobianOplusXj().data()[i]);
            }
            int np = bs->_Hpp->blockCols().size();
            W<int32_t>(np);
            for (int c = 0; c < np; c++) {
                W<int32_t>(bs->_Hpp->blockCols()[c].size());
                for (auto &kv : bs->_Hpp->blockCols()[c]) { W<int32_t>(kv.first); for (int i = 0; i < 49; i++) W(kv.second->data()[i]); }
            }
            for (int i = 0; i < np * 7; i++) W(bs->b()[i]);
            double lambda = alg->computeLambdaInit();
            W(lambda);
            bs->setLambda(lambda, true);
            bool ok = bs->solve();
            W<uint8_t>(ok);
            for (int i = 0; i < np * 7; i++) W(bs->x()[i]);
            delete opt;
        }
        // 전체 최적화(20 번), 반복마다 정점 값
        {
            g2o::OptimizationAlgorithmLevenberg *alg;
            auto *opt = make(vs, es, &alg);
            Rec rec; rec.opt = opt;
            for (auto &d : vs) rec.ids.push_back(d.id);
            opt->addPostIterationAction(&rec);
            opt->initializeOptimization();
            int it = opt->optimize(20);
            W<int32_t>(it); W<int32_t>(rec.it.size());
            for (auto &r : rec.it) for (double x : r) W(x);
            opt->removePostIterationAction(&rec);
            delete opt;
        }
    }
    fclose(F);
    return 0;
}
