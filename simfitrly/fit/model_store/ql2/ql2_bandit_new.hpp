
// Code generated by stanc v2.29.2
#include <stan/model/model_header.hpp>
namespace ql2_bandit_new_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 39> locations_array__ = 
{" (found before start of program)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 37, column 4 to column 33)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 38, column 4 to column 23)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 60, column 4 to column 18)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 61, column 4 to column 37)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 62, column 4 to column 15)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 65, column 4 to column 17)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 68, column 8 to column 73)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 70, column 8 to column 57)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 71, column 8 to column 46)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 67, column 33 to line 72, column 5)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 67, column 4 to line 72, column 5)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 42, column 4 to column 37)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 43, column 4 to column 15)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 45, column 4 to column 26)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 46, column 4 to column 26)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 52, column 8 to column 56)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 55, column 8 to column 57)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 56, column 8 to column 46)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 49, column 33 to line 57, column 5)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 49, column 4 to line 57, column 5)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 32, column 4 to column 20)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 33, column 19 to column 30)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 33, column 4 to column 32)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 34, column 19 to column 30)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 34, column 4 to column 32)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 9, column 8 to column 37)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 10, column 33 to column 43)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 10, column 8 to column 45)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 11, column 8 to column 24)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 12, column 8 to column 19)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 17, column 12 to column 36)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 20, column 12 to column 59)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 23, column 12 to column 55)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 24, column 12 to column 47)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 14, column 36 to line 25, column 9)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 14, column 8 to line 25, column 9)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 28, column 8 to column 45)",
 " (in '/home/stan/phd_thesis/simfitrly/fit/model_store/ql2/ql2_bandit_new.stan', line 8, column 6 to line 29, column 5)"};

struct likehood_ql2_functor__ {
  template <typename T0__, typename T1__,
            stan::require_stan_scalar_t<T0__>* = nullptr,
            stan::require_stan_scalar_t<T1__>* = nullptr>
  stan::promote_args_t<T0__, T1__>
  operator()(const T0__& alpha, const T1__& beta,
             const std::vector<int>& actions, const std::vector<int>& rewards,
             const int& trialCount, std::ostream* pstream__) const;
};

template <typename T0__, typename T1__,
          stan::require_stan_scalar_t<T0__>* = nullptr,
          stan::require_stan_scalar_t<T1__>* = nullptr>
  stan::promote_args_t<T0__, T1__>
  likehood_ql2(const T0__& alpha, const T1__& beta,
               const std::vector<int>& actions,
               const std::vector<int>& rewards, const int& trialCount,
               std::ostream* pstream__) {
    using local_scalar_t__ = stan::promote_args_t<T0__, T1__>;
    int current_statement__ = 0; 
    static constexpr bool propto__ = true;
    (void) propto__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      Eigen::Matrix<local_scalar_t__, 1, -1> Q =
         Eigen::Matrix<local_scalar_t__, 1, -1>::Constant(2, DUMMY_VAR__);
      current_statement__ = 26;
      stan::model::assign(Q, (Eigen::Matrix<double,1,-1>(2) << 0.5,
        0.5).finished(), "assigning variable Q");
      current_statement__ = 27;
      stan::math::validate_non_negative_index("choiceProbabilities",
                                              "trialCount", trialCount);
      std::vector<local_scalar_t__> choiceProbabilities =
         std::vector<local_scalar_t__>(trialCount, DUMMY_VAR__);
      Eigen::Matrix<local_scalar_t__, 1, -1> p =
         Eigen::Matrix<local_scalar_t__, 1, -1>::Constant(2, DUMMY_VAR__);
      local_scalar_t__ delta = DUMMY_VAR__;
      current_statement__ = 36;
      for (int trial = 1; trial <= trialCount; ++trial) {
        current_statement__ = 31;
        stan::model::assign(p,
          stan::math::transpose(
            stan::math::softmax(
              stan::math::multiply(stan::math::transpose(Q), beta))),
          "assigning variable p");
        current_statement__ = 32;
        stan::model::assign(choiceProbabilities,
          stan::model::rvalue(p, "p",
            stan::model::index_uni(stan::model::rvalue(actions, "actions",
                                     stan::model::index_uni(trial)))),
          "assigning variable choiceProbabilities", stan::model::index_uni(trial));
        current_statement__ = 33;
        delta = (stan::model::rvalue(rewards, "rewards",
                   stan::model::index_uni(trial)) -
                  stan::model::rvalue(Q, "Q",
                    stan::model::index_uni(stan::model::rvalue(actions,
                                             "actions",
                                             stan::model::index_uni(trial)))));
        current_statement__ = 34;
        stan::model::assign(Q,
          (stan::model::rvalue(Q, "Q",
             stan::model::index_uni(stan::model::rvalue(actions, "actions",
                                      stan::model::index_uni(trial)))) +
            (alpha * delta)),
          "assigning variable Q", stan::model::index_uni(stan::model::rvalue(
                                                           actions,
                                                           "actions",
                                                           stan::model::index_uni(trial))));
      }
      current_statement__ = 37;
      return stan::math::sum(stan::math::log(choiceProbabilities));
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    }
template <typename T0__, typename T1__, stan::require_stan_scalar_t<T0__>*,
          stan::require_stan_scalar_t<T1__>*>
stan::promote_args_t<T0__, T1__>
likehood_ql2_functor__::operator()(const T0__& alpha, const T1__& beta,
                                   const std::vector<int>& actions,
                                   const std::vector<int>& rewards,
                                   const int& trialCount,
                                   std::ostream* pstream__)  const
{
  return likehood_ql2(alpha, beta, actions, rewards, trialCount, pstream__);
}


class ql2_bandit_new_model final : public model_base_crtp<ql2_bandit_new_model> {

 private:
  int trial_count;
  std::vector<int> action_seq;
  std::vector<int> reward_seq; 
  
 
 public:
  ~ql2_bandit_new_model() { }
  
  inline std::string model_name() const final { return "ql2_bandit_new_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.29.2", "stancflags = "};
  }
  
  
  ql2_bandit_new_model(stan::io::var_context& context__,
                       unsigned int random_seed__ = 0,
                       std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "ql2_bandit_new_model_namespace::ql2_bandit_new_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 21;
      context__.validate_dims("data initialization","trial_count","int",
           std::vector<size_t>{});
      trial_count = std::numeric_limits<int>::min();
      
      
      current_statement__ = 21;
      trial_count = context__.vals_i("trial_count")[(1 - 1)];
      current_statement__ = 22;
      stan::math::validate_non_negative_index("action_seq", "trial_count",
                                              trial_count);
      current_statement__ = 23;
      context__.validate_dims("data initialization","action_seq","int",
           std::vector<size_t>{static_cast<size_t>(trial_count)});
      action_seq = 
        std::vector<int>(trial_count, std::numeric_limits<int>::min());
      
      
      current_statement__ = 23;
      action_seq = context__.vals_i("action_seq");
      current_statement__ = 24;
      stan::math::validate_non_negative_index("reward_seq", "trial_count",
                                              trial_count);
      current_statement__ = 25;
      context__.validate_dims("data initialization","reward_seq","int",
           std::vector<size_t>{static_cast<size_t>(trial_count)});
      reward_seq = 
        std::vector<int>(trial_count, std::numeric_limits<int>::min());
      
      
      current_statement__ = 25;
      reward_seq = context__.vals_i("reward_seq");
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = 1 + 1;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "ql2_bandit_new_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      local_scalar_t__ alpha = DUMMY_VAR__;
      current_statement__ = 1;
      alpha = in__.template read_constrain_lub<local_scalar_t__, jacobian__>(
                0, 1, lp__);
      local_scalar_t__ beta = DUMMY_VAR__;
      current_statement__ = 2;
      beta = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
               lp__);
      {
        Eigen::Matrix<local_scalar_t__, -1, 1> Q =
           Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(2, DUMMY_VAR__);
        current_statement__ = 12;
        stan::model::assign(Q, stan::math::rep_vector(0.0, 2),
          "assigning variable Q");
        local_scalar_t__ delta = DUMMY_VAR__;
        current_statement__ = 14;
        lp_accum__.add(stan::math::uniform_lpdf<propto__>(alpha, 0, 1));
        current_statement__ = 15;
        lp_accum__.add(stan::math::uniform_lpdf<propto__>(beta, 0, 50));
        current_statement__ = 20;
        for (int trial = 1; trial <= trial_count; ++trial) {
          current_statement__ = 16;
          lp_accum__.add(
            stan::math::categorical_logit_lpmf<propto__>(
              stan::model::rvalue(action_seq, "action_seq",
                stan::model::index_uni(trial)),
              stan::math::multiply(beta, Q)));
          current_statement__ = 17;
          delta = (stan::model::rvalue(reward_seq, "reward_seq",
                     stan::model::index_uni(trial)) -
                    stan::model::rvalue(Q, "Q",
                      stan::model::index_uni(stan::model::rvalue(action_seq,
                                               "action_seq",
                                               stan::model::index_uni(trial)))));
          current_statement__ = 18;
          stan::model::assign(Q,
            (stan::model::rvalue(Q, "Q",
               stan::model::index_uni(stan::model::rvalue(action_seq,
                                        "action_seq",
                                        stan::model::index_uni(trial)))) +
              (alpha * delta)),
            "assigning variable Q", stan::model::index_uni(stan::model::rvalue(
                                                             action_seq,
                                                             "action_seq",
                                                             stan::model::index_uni(trial))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_std_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "ql2_bandit_new_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      double alpha = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 1;
      alpha = in__.template read_constrain_lub<local_scalar_t__, jacobian__>(
                0, 1, lp__);
      double beta = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 2;
      beta = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
               lp__);
      out__.write(alpha);
      out__.write(beta);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      double log_like = std::numeric_limits<double>::quiet_NaN();
      Eigen::Matrix<double, -1, 1> Q =
         Eigen::Matrix<double, -1, 1>::Constant(2,
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 4;
      stan::model::assign(Q, stan::math::rep_vector(0.0, 2),
        "assigning variable Q");
      double delta = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 6;
      log_like = 0;
      current_statement__ = 11;
      for (int trial = 1; trial <= trial_count; ++trial) {
        current_statement__ = 7;
        log_like = (log_like +
                     stan::math::categorical_logit_lpmf<false>(
                       stan::model::rvalue(action_seq, "action_seq",
                         stan::model::index_uni(trial)),
                       stan::math::multiply(beta, Q)));
        current_statement__ = 8;
        delta = (stan::model::rvalue(reward_seq, "reward_seq",
                   stan::model::index_uni(trial)) -
                  stan::model::rvalue(Q, "Q",
                    stan::model::index_uni(stan::model::rvalue(action_seq,
                                             "action_seq",
                                             stan::model::index_uni(trial)))));
        current_statement__ = 9;
        stan::model::assign(Q,
          (stan::model::rvalue(Q, "Q",
             stan::model::index_uni(stan::model::rvalue(action_seq,
                                      "action_seq",
                                      stan::model::index_uni(trial)))) +
            (alpha * delta)),
          "assigning variable Q", stan::model::index_uni(stan::model::rvalue(
                                                           action_seq,
                                                           "action_seq",
                                                           stan::model::index_uni(trial))));
      }
      out__.write(log_like);
      out__.write(Q);
      out__.write(delta);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_std_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      local_scalar_t__ alpha = DUMMY_VAR__;
      alpha = in__.read<local_scalar_t__>();
      out__.write_free_lub(0, 1, alpha);
      local_scalar_t__ beta = DUMMY_VAR__;
      beta = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, beta);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"alpha", "beta", "log_like", "Q",
      "delta"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
      std::vector<size_t>{}, std::vector<size_t>{},
      std::vector<size_t>{static_cast<size_t>(2)}, std::vector<size_t>{
      }};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "alpha");
    param_names__.emplace_back(std::string() + "beta");
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      param_names__.emplace_back(std::string() + "log_like");
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "Q" + '.' + std::to_string(sym1__));
        } 
      }
      param_names__.emplace_back(std::string() + "delta");
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "alpha");
    param_names__.emplace_back(std::string() + "beta");
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      param_names__.emplace_back(std::string() + "log_like");
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "Q" + '.' + std::to_string(sym1__));
        } 
      }
      param_names__.emplace_back(std::string() + "delta");
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"alpha\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"log_like\",\"type\":{\"name\":\"real\"},\"block\":\"generated_quantities\"},{\"name\":\"Q\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"generated_quantities\"},{\"name\":\"delta\",\"type\":{\"name\":\"real\"},\"block\":\"generated_quantities\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"alpha\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"log_like\",\"type\":{\"name\":\"real\"},\"block\":\"generated_quantities\"},{\"name\":\"Q\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"generated_quantities\"},{\"name\":\"delta\",\"type\":{\"name\":\"real\"},\"block\":\"generated_quantities\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (1 + 1);
      const size_t num_transformed = 0;
      const size_t num_gen_quantities = 
  ((1 + 2) + 1);
      std::vector<double> vars_vec(num_params__
       + (emit_transformed_parameters * num_transformed)
       + (emit_generated_quantities * num_gen_quantities));
      std::vector<int> params_i;
      write_array_impl(base_rng, params_r, params_i, vars_vec,
          emit_transformed_parameters, emit_generated_quantities, pstream);
      vars = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        vars_vec.data(), vars_vec.size());
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (1 + 1);
      const size_t num_transformed = 0;
      const size_t num_gen_quantities = 
  ((1 + 2) + 1);
      vars.resize(num_params__
        + (emit_transformed_parameters * num_transformed)
        + (emit_generated_quantities * num_gen_quantities));
      write_array_impl(base_rng, params_r, params_i, vars, emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 2> names__{"alpha", "beta"};
      const std::array<Eigen::Index, 2> constrain_param_sizes__{1, 1};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
     }; } 
using stan_model = ql2_bandit_new_model_namespace::ql2_bandit_new_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return ql2_bandit_new_model_namespace::profiles__;
}

#endif


