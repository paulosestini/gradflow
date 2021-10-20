from gradflow import Node

x = Node(2)
y = Node(3)

z = 3*(x**2 + y)
# dz/dx = 6x -> dz/dx=6*2=12
# dz/dy = 3

z.backward()
print("Primeiro caso: ")
print(z.grad, x.grad, y.grad)

z.zero_grad()
z = Node.log(x) + 2*x + y**3
# dz/dx = 1/x + 2 = 1/2 + 2 = 2,5
# dz/dy = 3*(y**2) = 3 * 9 = 27

z.backward()
print("Segundo caso: ")
print(z.grad, x.grad, y.grad)