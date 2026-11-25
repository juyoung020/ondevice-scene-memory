//! ORB-SLAM3 지역 번들 조정(Optimizer::LocalBundleAdjustment)이 부르는 g2o 최적화를 비트 단위로 옮긴 것.
//! 원본: UZ-SLAMLab/ORB_SLAM3(GPL-3.0) src/Optimizer.cc + Thirdparty/g2o(BSD) — SparseOptimizer · OptimizationAlgorithmLevenberg ·
//! BlockSolver<6,3>(슈어 소거) · LinearSolverEigen(Eigen 3.4 SimplicialLDLT + AMD) · 간선 3 종(edges.rs).
//! 비교 기준: -ffp-contract=off 로 다시 빌드한 원본(docs/orb3_port.md 공통 규칙). 컴파일러 축약은 없고,
//! Eigen 이 묶음 곱(pmadd)에서 vfmadd 내장 함수를 직접 쓰는 자리만 한 번 반올림(`se3::fused`)으로 남는다 —
//! 고정 크기 행렬 곱의 묶음 행(3 행이면 0·1 행, 6 행이면 전부), 나머지(낱개 행 · 내적 · 원소별 식)는 소스 순서 그대로.
//! 저장소 밖 하네스(~/opt/orb3_ba_bg)로 덤프한 실제 호출과 비트 비교한다.
//!
//! 쓰는 법: `Problem` 에 정점(id 순서와 무관) · 간선(원본이 addEdge 한 순서)을 넣고 `optimize(10)`.
pub mod edges;
pub mod gba;
pub mod lba;
pub mod ldlt;
pub mod mlba;
pub mod se3;
pub mod sim3;
pub mod solver;

pub use edges::{Edge, Kind};
pub use solver::{optimize, optimize_stop, Outcome};

/// 정점: 자세(SE3Quat: 사원수 [x,y,z,w] · 이동) 또는 점(VertexSBAPointXYZ, 주변화)
#[derive(Clone, Debug)]
pub enum Est {
    Pose { q: [f64; 4], t: [f64; 3] },
    Point([f64; 3]),
}

#[derive(Clone, Debug)]
pub struct Vertex {
    pub id: i32,
    pub fixed: bool,
    pub est: Est,
}

/// g2o 그래프 하나. lambda_init > 0 이면 setUserLambdaInit(관성 지도일 때 100)
#[derive(Clone, Debug, Default)]
pub struct Problem {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub lambda_init: f64,
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod lba_tests;
#[cfg(test)]
mod gba_tests;
