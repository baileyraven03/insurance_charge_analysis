import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

insurance = pd.read_csv(
    'Learning/Codecademy Projects/Data Scientist - Machine Learning Specialist/Medical_insurance_full_project/insurance.csv')
print(insurance.head(3))

#  Age analysis of the data
ages = np.array(insurance.age)
# Finding max, min and mean values
max_age = ages.max()
min_age = ages.min()
average_age = np.mean(ages)
# Printing gender findings
print('The maximum age of payees is {} years old.'.format(max_age))
print('The minimum age of payees is {} years old.'.format(min_age))
print('The average age of payees is {:.4} years old.'.format(average_age))


plt.scatter(x=insurance.age, y=insurance.charges)
plt.plot([3, 70], [1000, 100000], color='red')
plt.title("Charges per age ($/Year)")
plt.xlabel('Age (in years)')
plt.ylabel('Charges accrude ($)')
plt.show()
plt.close()

#  Gender analysis of the data
men = insurance[(insurance.sex == 'male')]
women = insurance[(insurance.sex == 'female')]
# Finding max and min values for ages
max_men = np.max(men.age)
max_women = np.max(women.age)
min_men = np.min(men.age)
min_women = np.min(women.age)
# Finding mean, mode and median values for ages
median_men = np.median(men.age)
mode_men = men.age.mode()
mean_men = men.age.mean()
median_women = np.median(women.age)
mode_women = women.age.mode()
mean_women = women.age.mean()

print("The maximum age of males in the group is {} years old and {} years old for women.".format(
    max_men, max_women))
print("The minimum age of males in the group is {} years old and {} years old for women.".format(
    min_men, min_women))
print("For males in the group:")
print("Mean: {:.2f}".format(mean_men))
print("Median: {:.2f}".format(median_men))
print("Mode: " + str(mode_men))
print("For females in the group:")
print("Mean: {:.2f}".format(mean_women))
print("Median: {:.2f}".format(median_women))
print("Mode: " + str(mode_women[1]))

sns.boxplot(x='sex', y='charges', data=insurance)
plt.title("Age by gender of payees")
plt.ylabel('Age (in years)')
plt.xlabel('Gender')
plt.show()
plt.close()

# Smoking analysis for the data
smokers = insurance[(insurance.smoker == 'yes')]
non_smokers = insurance[(insurance.smoker == 'no')]
smokers_mean_cost = np.mean(smokers.charges)
non_smokers_mean_cost = np.mean(non_smokers.charges)
smokers_variance = insurance[(insurance.smoker == 'yes')].charges.var()
non_smokers_variance = non_smokers.charges.var()
smokers_std = smokers.charges.std()
non_smokers_std = np.std(non_smokers.charges)

print("Smokers account for an average cost of ${:.2f}, and non-smokers have an average cost of ${:.2f}".format(
    smokers_mean_cost, non_smokers_mean_cost))
print("This is an average cost of ${:.2f} grater for smokers comapred to non-smokers".format(
    smokers_mean_cost - non_smokers_mean_cost))
print("Smokers have a variance of ${:.2f}, where as non-smokers have a variance of ${:.2f}.".format(
    smokers_variance, non_smokers_variance))
print("This means smokers have a standard deviation of ${:.2f}, where as non-smokers have a standard deviation of ${:.6}.".format(
    smokers_std, non_smokers_std))

smoker_and_sex = insurance.groupby(['smoker', 'sex'])[
    'charges'].mean().reset_index()
pivot_smosex = smoker_and_sex.pivot(columns='smoker',
                                    index='sex',
                                    values='charges').reset_index()
print("The average charge of smokers by gender:")
print(pivot_smosex)

contingency_smochi = pd.crosstab(insurance.children, insurance.smoker)
print("The number of smokers by number of children:")
print(contingency_smochi)

sns.boxplot(x='smoker', y='charges', data=insurance)
plt.title("Charges for smokers & non-smokers")
plt.ylabel('Age (in years)')
plt.xlabel('Gender')
plt.show()
plt.close()

print(insurance.describe(include='all'))
