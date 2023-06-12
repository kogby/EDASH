import pandas as pd
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import dblquad
from dataclasses import dataclass
from scipy.linalg import det
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import r2_score
import scipy.stats as stats

@dataclass
class continuous:
    dependent_var: str = None

    def _gaussion(self, x, data_list):
        mean = np.mean(data_list)
        std = np.std(data_list) 
        return 1/(np.sqrt(2*np.pi)*std)*np.exp(-1/2*((x-mean)/std)**2)

    def _pearson_corr_coefficient(self, data_list_1, data_list_2):
        mu_x = np.mean(data_list_1)
        mu_y = np.mean(data_list_2) 
        std_x = np.std(data_list_1)
        std_y = np.std(data_list_2)
        # cov_x_y = sum((a - mu_x) * (b - mu_y) for (a,b) in zip(data_list_1,data_list_2)) / len(data_list_1)
        return sum((a - mu_x) * (b - mu_y) for (a,b) in zip(data_list_1,data_list_2)) / len(data_list_1)/(std_x*std_y), mu_x, mu_y, std_x, std_y

    
    def _joint_gaussion(self, x, y, data_list_1, data_list_2):
        rho, mu_x, mu_y, std_x, std_y  = self._pearson_corr_coefficient(data_list_1, data_list_2)
        x_diff = x-mu_x
        y_diff = y-mu_y
        S = (x_diff/std_x)**2 + (y_diff/std_y)**2 - 2*rho*x_diff*y_diff/(std_x* std_y)
        return 1/(2*np.pi*std_x*std_y*np.sqrt(1 - rho**2))*np.exp(-S/(2*(1-rho**2)))
         
    def entropy(self, data_list):
        # result, _ = integrate.quad(lambda x: self._gaussion(x, data_list)*np.log(1/self._gaussion(x, data_list)), -30,30)
        return 1/2*np.log(2*np.pi*np.e*np.std(data_list)**2)

    def relative_entropy(self, data_list_1, data_list_2):
        std_1 = np.std(data_list_1)
        std_2 = np.std(data_list_2)
        # result, _ = integrate.quad(lambda x: self._gaussion(x, data_list_1)* np.log(self._gaussion(x, data_list_1)/self._gaussion(x, data_list_2)), -30,30)
        return 1/2*(np.log(std_2**2/std_1**2) + std_1**2/std_2**2 + (np.mean(data_list_1)-np.mean(data_list_2))**2/std_2**2 -1)

    def js_entropy(self, data_list_1, data_list_2):
        return 1/2*(self.relative_entropy(data_list_1, data_list_2)+ self.relative_entropy(data_list_2, data_list_1)), self.relative_entropy(data_list_1, data_list_2), self.relative_entropy(data_list_2, data_list_1)

    def joint_entropy_2(self, data_list_1, data_list_2):
        ''' 
        entropy under both consideration
        '''
        # f = lambda x, y: -self._joint_gaussion(x, y, data_list_1, data_list_2)*np.log(self._joint_gaussion(x, y, data_list_1, data_list_2))
        # val, _ = integrate.dblquad(f, -20, 20, -20, 20)
        # return val
        if list(data_list_1) == list(data_list_2):
            return self.entropy(data_list_1)
        n = 2
        cov_matrix = np.cov(data_list_1, data_list_2)
        return (1/2)*(n*np.log(2*np.pi) + n + np.log(det(cov_matrix)))

    def mutual_information(self, data_list_1, data_list_2):
        ''' 
        dependence of data_list_1 and data_list_2 
        '''
        H_x = self.entropy(data_list_1)
        H_y = self.entropy(data_list_2)
        H_xy = self.joint_entropy_2(data_list_1, data_list_2)
        return H_x + H_y - H_xy

    def mi_matrix(self, df, table=True, plot=True, title=''):
        """
        Calculates the mutual information matrix for a given dataframe.
        
        Parameters:
        df (pandas.DataFrame): The dataframe to calculate the mutual information matrix from.
        
        Returns:
        pandas.DataFrame: The mutual information matrix.
        """
        mi_mat = pd.DataFrame(index=df.columns, columns=df.columns)
        for col1 in df.columns:
            for col2 in df.columns:
                mi_mat.loc[col1, col2] = self.mutual_information(df[col1], df[col2])
        # plot
        if plot:
            plt.figure(figsize=(8, 8))
            uniform_data = np.array(mi_mat).astype(float)
            df_masked = mi_mat.mask(np.tri(mi_mat.shape[0], dtype=bool))
            vmax = df_masked.max().max()
            vmin = df_masked.min().min()
            ax = sns.heatmap(uniform_data, linewidth=0.1, vmin=vmin, vmax=vmax, annot=True)
            title = "MI " + title
            plt.title(title)
            plt.show()
        # dataframe
        if table:
            return mi_mat

    def basic_info(self, col_list):
        # mean, var, corr_matrix, Q1-Q3 of each column
        min = np.mean(col_list)
        max = np.max(col_list)
        avg = np.mean(col_list)
        q1 = np.percentile(col_list, 25)
        q3 = np.percentile(col_list, 75)
        std = np.std(col_list)   
        skewness = stats.skew(col_list)
        kurtosis = stats.kurtosis(col_list)
        return min, max, avg, q1, q3, std, skewness, kurtosis

    def compare_base(self, df, method = "zero"):
        "when need to compare with missing data, consider drop all missing value or take the case as 0"
        if method == 'drop':
            return df.copy().dropna()
        elif method == "zero":  
            return df.copy().fillna(0,inplace=True)

    def pair_plot(self, df_before, df_after):
        da, db = df_after.copy(), df_before.copy()
        da['**impute**'] = True
        db['**impute**'] = False
        combined_df = pd.concat([db, da], ignore_index=True)
        # sns.pairplot(combined_df, kind='reg', hue='**impute**')
        sns.set(style='ticks')
        plot = sns.pairplot(combined_df, kind='reg', hue='**impute**', diag_kind='hist', diag_kws={'alpha': 0.5}, plot_kws={'scatter_kws': {'alpha': 0.3}})
        plt.show()


    def comparison(self, df_before, df_after, method='all'):
        """
        Compare some metrics to show difference percentage of imputed before/after
        """
        # entropy_pair_list = []
        # js_entropy_list = []
        df_entropy = {}
        df_js_entropy = {}
        df_basic_before = {}
        df_basic_after = {}

        # for entropy
        for col in df_before.columns:
            entropy_before, entropy_after = self.entropy(df_before[col]), self.entropy(df_after[col])
            min_before, max_before, avg_before, q1_before, q3_before, std_before, skewness_before, kurtosis_before = self.basic_info(df_before[col])
            min_after, max_after, avg_after, q1_after, q3_after, std_after, skewness_after, kurtosis_after = self.basic_info(df_after[col])
            df_entropy.update({col:[entropy_before, entropy_after, self.percentage_difference(entropy_before, entropy_after)]})
            js, kl_12, kl_21 = self.js_entropy(df_before[col], df_after[col])
            df_js_entropy.update({col: [js, kl_12, kl_21]})
            df_basic_before.update({col:[min_before, max_before, avg_before, q1_before, q3_before, std_before, skewness_before, kurtosis_before]})
            df_basic_after.update({col:[min_after, max_after, avg_after, q1_after, q3_after, std_after, skewness_after, kurtosis_after]})

        df_entropy = pd.DataFrame(df_entropy, index=['entropy_before', 'entropy_after', 'percentage_difference'])
        df_entropy.columns.name = 'Entropy'

        df_js_entropy = pd.DataFrame(df_js_entropy, index=['js_entropy', 'KL(1||2)', 'KL(2||1)'])
        df_js_entropy.columns.name = 'KL-Diverge'

        df_basic_before = pd.DataFrame(df_basic_before, index=['min', 'max', 'avg', 'q1', 'q3', 'std', 'skewness', 'kurtosis'])
        df_basic_after = pd.DataFrame(df_basic_after, index=['min', 'max', 'avg', 'q1', 'q3', 'std', 'skewness', 'kurtosis'])
        df_basic_before.columns.name = 'Basic Info Before'
        df_basic_after.columns.name = 'Basic Info After'



        print('\n*** Entropy ***')
        display(df_entropy)

        print('\n*** KL Divergence(Relatice Entropy) ***')
        display(df_js_entropy)

        print('\n*** Basic Info ***')
        display(df_basic_before)
        display(df_basic_after)

        print('\n*** Mutual Informtaion Matrix ***')
        self.mi_matrix(df_before, table=False, plot=True, title='before')
        self.mi_matrix(df_after, table=False, plot=True, title='after')

        print('\n*** Covarience ***')
        # 繪製heatmap
        plt.figure(figsize=(8, 8))
        cov_mat = df_before.cov()
        ax = sns.heatmap(cov_mat, annot=True, cmap='coolwarm')
        plt.title('Covariance Matrix Before')
        plt.show()

        plt.figure(figsize=(8, 8))
        cov_mat_2 = df_after.cov()
        ax = sns.heatmap(cov_mat_2, annot=True, cmap='coolwarm')
        plt.title('Covariance Matrix After')
        plt.show()

        print('\n*** Pairplot ***')
        self.pair_plot(df_before, df_after)

    def percentage_difference(self, a, b):
        return (b-a)/a
    def r2(y_true,y_pred):
        '''
        R Square
        '''
        return r2_score(y_true,y_pred)