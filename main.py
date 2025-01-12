from graph import TravelGraph
import asyncio

async def main():
    cities = ['Paris', 'Berlin', 'London', 'Madrid', 'Rome']

    graph = TravelGraph()
    await graph.create(cities)

    print(graph.cities)
    print()
    print(graph.routes)

    print()

    print(graph.routes['Paris'])

if __name__ == '__main__':
    asyncio.run(main())
